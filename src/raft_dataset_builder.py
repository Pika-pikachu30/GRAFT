import json
import logging
import random
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import networkx as nx
from utils import setup_logger, LLMWrapper
from graph_indexer import CommunitySummary

logger = setup_logger(__name__)

@dataclass
class Question:
    id: str
    text: str
    type: str  # local, global, multihop, comparative
    source_community: str

@dataclass
class ContextDocument:
    id: str
    text: str
    is_oracle: bool
    source: str # e.g., 'community_C0_1'

@dataclass
class CoTAnswer:
    relevant_docs: List[str]
    reasoning: str
    final_answer: str
    citations: List[str]

@dataclass
class TrainingSample:
    instruction: str
    input: str
    output: str

class RAFTDatasetBuilder:
    """
    Handles Phase 2 of GRAFT: Training Data Generation.
    Generates graph-aware RAFT fine-tuning data.
    """
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        self.llm = llm_wrapper or LLMWrapper()

    def generate_questions(self, community_summaries: Dict[str, CommunitySummary], n_questions_per_community: int = 5) -> List[Question]:
        """Generates diverse questions requiring local/global reasoning based on community summaries."""
        questions = []
        for comm_id, summary in community_summaries.items():
            prompt = (
                f"Given the following community summary from a knowledge graph, generate {n_questions_per_community} "
                f"diverse questions that would require understanding this information.\n"
                f"Include a mix of 'local' (entity lookup), 'global' (sensemaking), 'multihop' (connections), and 'comparative' questions.\n"
                f"Format output exactly as:\n[TYPE] Question text\n\n"
                f"Example:\n[local] What is Entity X?\n[global] What are the main themes of this community?\n\n"
                f"Summary:\nTitle: {summary.title}\n{summary.summary}\n"
            )
            
            response = self.llm.generate(prompt=prompt)
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("["):
                    end_bracket = line.find("]")
                    if end_bracket != -1:
                        q_type = line[1:end_bracket].strip().lower()
                        q_text = line[end_bracket+1:].strip()
                        questions.append(Question(
                            id=str(uuid.uuid4()),
                            text=q_text,
                            type=q_type,
                            source_community=comm_id
                        ))
        logger.info(f"Generated {len(questions)} questions from {len(community_summaries)} communities.")
        return questions

    def build_oracle_context(self, question: Question, community_summaries: Dict[str, CommunitySummary], graph: nx.Graph) -> List[ContextDocument]:
        """Builds the oracle context required to answer the question."""
        oracle_docs = []
        
        # In a full BM25 + Cosine similarity implementation, we'd rank all summaries.
        # For simplicity, we just use the known source community and its immediate neighbors/sub-communities.
        source_comm = community_summaries.get(question.source_community)
        if source_comm:
            oracle_docs.append(ContextDocument(
                id=f"doc_{question.source_community}",
                text=f"Title: {source_comm.title}\n{source_comm.summary}",
                is_oracle=True,
                source=question.source_community
            ))
            
        # Add local subgraph context if it's a local/multihop question
        if question.type in ["local", "multihop"]:
            subgraph_nodes = source_comm.nodes[:10] if source_comm else [] # Limit subgraph context
            if subgraph_nodes and graph.number_of_nodes() > 0:
                valid_nodes = [n for n in subgraph_nodes if n in graph]
                sub = graph.subgraph(valid_nodes)
                lines = [f"Entity: {n}" for n in sub.nodes()]
                edges = [f"{u} -> {v}" for u, v in sub.edges()]
                oracle_docs.append(ContextDocument(
                    id=f"graph_{question.source_community}",
                    text="Local Graph Context:\n" + "\n".join(lines + edges),
                    is_oracle=True,
                    source=f"graph_{question.source_community}"
                ))
                
        return oracle_docs

    def inject_distractor_documents(self, oracle_context: List[ContextDocument], all_summaries: Dict[str, CommunitySummary], n_distractors: int = 3) -> List[ContextDocument]:
        """Injects random distractor documents to teach the model to ignore irrelevant context."""
        augmented_context = list(oracle_context)
        oracle_ids = {doc.source for doc in oracle_context}
        
        available_distractors = [k for k in all_summaries.keys() if k not in oracle_ids]
        if len(available_distractors) > n_distractors:
            distractor_ids = random.sample(available_distractors, n_distractors)
        else:
            distractor_ids = available_distractors
            
        for d_id in distractor_ids:
            summary = all_summaries[d_id]
            augmented_context.append(ContextDocument(
                id=f"distractor_{d_id}",
                text=f"Title: {summary.title}\n{summary.summary}",
                is_oracle=False,
                source=d_id
            ))
            
        random.shuffle(augmented_context)
        return augmented_context

    def generate_cot_answer(self, question: Question, oracle_context: List[ContextDocument]) -> CoTAnswer:
        """Prompts LLM with only the oracle context to get the gold standard CoT and citations."""
        context_text = ""
        for i, doc in enumerate(oracle_context):
            context_text += f"\n--- Document {i+1} [{doc.source}] ---\n{doc.text}\n"

        prompt = (
            f"You are an expert AI answering a question based strictly on the provided context.\n"
            f"Please output exactly in this format:\n"
            f"RELEVANT DOCS: <comma separated document sources>\n"
            f"REASONING: <step-by-step thinking process>\n"
            f"ANSWER: <final answer>\n"
            f"CITATIONS: <comma separated citations like [Community: C0_1]>\n\n"
            f"Context:{context_text}\n\n"
            f"Question: {question.text}\n"
        )
        
        response = self.llm.generate(prompt=prompt)
        
        # Parse response
        rel_docs, reasoning, answer, citations = [], "", "", []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("RELEVANT DOCS:"):
                rel_docs = [x.strip() for x in line[len("RELEVANT DOCS:"):].split(",") if x.strip()]
            elif line.startswith("REASONING:"):
                reasoning = line[len("REASONING:"):].strip()
            elif line.startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()
            elif line.startswith("CITATIONS:"):
                citations = [x.strip() for x in line[len("CITATIONS:"):].split(",") if x.strip()]
                
        # If parsing fails on multiline, fallback
        if not answer:
            answer = response
            
        return CoTAnswer(
            relevant_docs=list(set(rel_docs + [d.source for d in oracle_context])),
            reasoning=reasoning or "Deduced from provided context.",
            final_answer=answer,
            citations=citations or [f"[Community: {oracle_context[0].source}]" if oracle_context else "None"]
        )

    def build_training_sample(self, question: Question, augmented_context: List[ContextDocument], cot_answer: CoTAnswer) -> TrainingSample:
        """Formats the context, question, and CoT answer into an instruction tuning sample."""
        input_text = "Context Documents:\n"
        for i, doc in enumerate(augmented_context):
            input_text += f"\n<doc id='{doc.source}'>\n{doc.text}\n</doc>\n"
            
        output_text = (
            f"Let's think step by step.\n"
            f"{cot_answer.reasoning}\n\n"
            f"Based on this reasoning, the answer is:\n{cot_answer.final_answer}\n\n"
            f"Citations: {', '.join(cot_answer.citations)}\n"
        )
        
        return TrainingSample(
            instruction=f"Answer the following question using the provided context documents. Cite the context where appropriate.\nQuestion: {question.text}",
            input=input_text,
            output=output_text
        )

    def export_dataset(self, samples: List[TrainingSample], output_path: str, format: str = "jsonl"):
        """Exports the generated dataset, splitting into train/val/test."""
        random.shuffle(samples)
        total = len(samples)
        
        if total == 0:
            logger.warning("No samples to export.")
            return
            
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:]
        }
        
        import os
        os.makedirs(output_path, exist_ok=True)
        
        for split_name, split_data in splits.items():
            filepath = os.path.join(output_path, f"{split_name}.{format}")
            with open(filepath, "w", encoding="utf-8") as f:
                for sample in split_data:
                    if format == "jsonl":
                        f.write(json.dumps(asdict(sample)) + "\n")
                        
        logger.info(f"Exported {total} samples to {output_path} (train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}).")
