import torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import setup_logger, LLMWrapper
import networkx as nx
from typing import Dict


from graph_indexer import CommunitySummary

logger = setup_logger(__name__)

@dataclass
class QueryType:
    category: str  # local, global, multihop, comparative

@dataclass
class Context:
    documents: List[str]
    source_ids: List[str]

@dataclass
class GRAFTAnswer:
    relevant_docs: List[str]
    reasoning_chain: str
    final_answer: str
    citations: List[str]

class GRAFTInference:
    """
    Handles Phase 4 of GRAFT: Inference Engine.
    Executes queries against a fine-tuned GRAFT model using graph context.
    """
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        self.model = None
        self.tokenizer = None
        self.base_llm = llm_wrapper or LLMWrapper()  # Base LLM used for heuristics (like query classification)

    def load_model(self, model_path: str, quantize: bool = True):
        """Loads a fine-tuned LoRA-merged HuggingFace model."""
        import torch
        from transformers import BitsAndBytesConfig
        logger.info(f"Loading GRAFT Inference model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if quantize:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
            
        self.model.eval()
        logger.info("Inference model loaded.")

    def classify_query(self, query: str) -> QueryType:
        """Classifies the given user query to branch routing strategy."""
        prompt = (
            f"Classify the following question into exactly one category: 'local', 'global', 'multihop', or 'comparative'.\n"
            f"Question: {query}\n"
            f"Output only the category name.\n"
        )
        response = self.base_llm.generate(prompt=prompt).strip().lower()
        if "local" in response: category = "local"
        elif "multihop" in response: category = "multihop"
        elif "comparative" in response: category = "comparative"
        else: category = "global"
            
        logger.debug(f"Classified query '{query}' as {category}.")
        return QueryType(category=category)

    def retrieve_context(self, query: str, community_summaries: Dict[str, CommunitySummary], graph: nx.Graph, query_type: QueryType) -> Context:
        """Retrieves and ranks context based on question classification."""
        # A full map-reduce semantic search would typically use embeddings.
        # This implementation provides a simplified mock of semantic retrieval using the graph.
        
        docs = []
        sources = []
        
        if query_type.category == "global":
            # Global queries use root summaries (like Map-Reduce in GraphRAG)
            root_comms = [c for c in community_summaries.values() if c.level == 0]
            # Mock retrieving top 3 by length (in reality, cosine similarity)
            root_comms.sort(key=lambda x: x.tokens, reverse=True)
            for c in root_comms[:3]:
                docs.append(f"Title: {c.title}\n{c.summary}")
                sources.append(f"C0_{c.level}_{hash(c.title)}")
                
        elif query_type.category in ["local", "multihop"]:
            # Use sentence embeddings for semantic extraction
            try:
                from sentence_transformers import SentenceTransformer, util
                if not hasattr(self, 'embedder'):
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self.embedder = None
            
            best_match = None
            max_score = -1.0
            
            if self.embedder is not None:
                q_emb = self.embedder.encode(query, convert_to_tensor=True)
                for c_id, summary in community_summaries.items():
                    c_emb = self.embedder.encode(summary.summary, convert_to_tensor=True)
                    score = util.cos_sim(q_emb, c_emb).item()
                    if score > max_score:
                        max_score = score
                        best_match = summary
            else:
                # Fallback to token overlap if sentence-transformers not found
                q_words = set(query.lower().split())
                for c_id, summary in community_summaries.items():
                    overlap = len(set(summary.summary.lower().split()).intersection(q_words))
                    if overlap > max_score:
                        max_score = overlap
                        best_match = summary
                    
            if best_match:
                docs.append(f"Title: {best_match.title}\n{best_match.summary}")
                sources.append("C_BestMatch")
                # Also inject local neighbor graph
                if best_match.nodes:
                    valid_nodes = [n for n in best_match.nodes[:5] if n in graph]
                    sub = graph.subgraph(valid_nodes)
                    graph_txt = "\n".join([f"Entity: {n}" for n in sub.nodes()]) + "\n".join([f"{u}->{v}" for u,v in sub.edges()])
                    docs.append("Local Graph:\n" + graph_txt)
                    sources.append("LocalSubgraph")
        else:
            # Comparative mapping
            docs.append("Context retrieval strategy for comparative questions.")
            sources.append("Comp")

        logger.debug(f"Retrieved {len(docs)} context documents.")
        return Context(documents=docs, source_ids=sources)

    def generate_answer(self, query: str, context: Context) -> GRAFTAnswer:
        """Generates the grounded CoT answer using the loaded fine-tuned model."""
        # If no fine-tuned model loaded, fall back to base LLM
        if self.model is None or self.tokenizer is None:
            ctx_text = "\n".join(context.documents)
            prompt = (
                f"Answer this question using the context below.\n"
                f"Question: {query}\n\nContext:\n{ctx_text}\n\n"
                f"Think step by step then give your answer."
            )
            response = self.base_llm.generate(prompt=prompt)
            return GRAFTAnswer(
                relevant_docs=context.source_ids,
                reasoning_chain="Answered via base LLM fallback.",
                final_answer=response,
                citations=context.source_ids[:2]
            )

        input_text = "Context Documents:\n"
        for idx, doc in enumerate(context.documents):
            source_id = context.source_ids[idx]
            input_text += f"\n<doc id='{source_id}'>\n{doc}\n</doc>\n"
            
        full_prompt = f"### Instruction:\nAnswer the following question using the provided context documents. Cite the context where appropriate.\nQuestion: {query}\n\n### Input:\n{input_text}\n\n### Response:\nLet's think step by step.\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        try:
            # Parse the model's generated output, looking for reasoning and citations based on fine-tuning distribution.
            reasoning, final, cites = "", "", []
            
            parts = generated_text.split("Based on this reasoning, the answer is:")
            if len(parts) == 2:
                reasoning = parts[0].strip()
                ans_cites = parts[1].split("Citations:")
                final = ans_cites[0].strip()
                cites = [c.strip() for c in ans_cites[1].split(",")] if len(ans_cites) > 1 else []
            else:
                final = generated_text.strip()
                
            return GRAFTAnswer(
                relevant_docs=context.source_ids,
                reasoning_chain=reasoning or "No reasoning extracted.",
                final_answer=final,
                citations=cites
            )
        except Exception as e:
            logger.error(f"Error parsing GRAFT generated answer: {e}")
            return GRAFTAnswer(relevant_docs=[], reasoning_chain="", final_answer=generated_text, citations=[])

    def batch_inference(self, queries: List[str], community_summaries: Dict[str, CommunitySummary], graph: nx.Graph) -> List[GRAFTAnswer]:
        """Parallelized batch inference using ThreadPoolExecutor."""
        import concurrent.futures
        
        def process_query(q):
            q_type = self.classify_query(q)
            ctx = self.retrieve_context(q, community_summaries, graph, q_type)
            return self.generate_answer(q, ctx)
            
        answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {executor.submit(process_query, q): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                answers.append(future.result())
                
        return answers
