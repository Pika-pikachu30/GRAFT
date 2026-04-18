import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple
from typing import Optional


import numpy as np
from datasets import Dataset
from rouge_score import rouge_scorer
import bert_score
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy

from utils import setup_logger, LLMWrapper
from data_loader import EvalQuestion
from graft_inference import GRAFTAnswer

logger = setup_logger(__name__)

class Evaluator:
    """
    Handles Phase 3 of GRAFT: Comprehensive Evaluation Framework.
    Implements LLM-as-Judge, Claim-Based Metrics, Standard NLP Metrics, Faithfulness, and Efficiency.
    """

    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None, output_dir: str = "results"):
        self.llm = llm_wrapper or LLMWrapper()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results: Dict[str, Any] = {}

    def llm_pairwise_judge(self, answers_a: List[str], answers_b: List[str], questions: List[str], metrics: List[str] = ["Comprehensiveness", "Diversity", "Empowerment", "Directness"]) -> Dict[str, float]:
        """Runs LLM-as-Judge pairwise win-rate comparisons for a set of answers."""
        if len(answers_a) != len(answers_b) or len(answers_a) != len(questions):
            raise ValueError("Mismatched list lengths for pairwise judge.")

        logger.info(f"Running LLM-as-Judge pairwise comparisons for {len(questions)} questions on {metrics}...")
        
        # A simple prompt setup mimicking GraphRAG paper Algorithm 1
        prompt_template = (
            "You are an expert evaluator. Compare Answer A and Answer B to the following Question based strictly on {metric}.\n"
            "Question: {question}\n\n"
            "Answer A: {ans_a}\n"
            "Answer B: {ans_b}\n\n"
            "Which answer is better according to the metric? Output exactly 'A', 'B', or 'TIE'."
        )

        win_rates = {m: {'A_wins': 0, 'B_wins': 0, 'Ties': 0} for m in metrics}
        
        for idx, (q, a, b) in enumerate(zip(questions, answers_a, answers_b)):
            # In a real implementation, you'd run multiple replicates (e.g. 5) of this per question to reduce LLM variance.
            for m in metrics:
                prompt = prompt_template.format(metric=m, question=q, ans_a=a, ans_b=b)
                response = self.llm.generate(prompt=prompt).strip().upper()
                
                # Randomize order internally to avoid position bias (omitted for brevity, assume LLM outputs A or B reliably)
                if response.startswith("A"): win_rates[m]['A_wins'] += 1
                elif response.startswith("B"): win_rates[m]['B_wins'] += 1
                else: win_rates[m]['Ties'] += 1
                
        # Calculate win rates excluding ties
        final_win_rates = {}
        for m, stats in win_rates.items():
            total = stats['A_wins'] + stats['B_wins']
            final_win_rates[m] = (stats['A_wins'] / total) if total > 0 else 0.5
            
        logger.info(f"Pairwise win-rates calculated: {final_win_rates}")
        return final_win_rates

    def claim_metrics(self, graft_answers: List[GRAFTAnswer], gold_answers: List[str], distance_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8]) -> Dict[str, Any]:
        """Calculates Comprehensiveness, Diversity (Agglomerative Clustering), and Citation Precision."""
        logger.info("Computing claim-based metrics (GraphRAG Experiment 2)...")
        
        avg_claims = 0.0
        citation_precision = 0.0
        diversity_scores = {th: 0.0 for th in distance_thresholds}
        
        # Simulating agglomerative clustering using sklearn & rouge distance
        from sklearn.cluster import AgglomerativeClustering
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        for g_ans in graft_answers:
            # 1. Comprehensiveness (Mock: Number of sentences treated as claims)
            claims = [str(x) for x in g_ans.final_answer.split(". ") if len(x) > 5]
            num_claims = len(claims)
            avg_claims += num_claims
            
            # 2. Citation Precision
            valid_cites = sum(1 for c in g_ans.citations if any(r in c for r in g_ans.relevant_docs))
            precision = (valid_cites / len(g_ans.citations)) if g_ans.citations else 0.0
            citation_precision += precision
            
            # 3. Diversity (Cluster Count based on ROUGE-L distance map)
            if num_claims > 1:
                # Build distance matrix (1 - ROUGE-L f1)
                dist_matrix = np.zeros((num_claims, num_claims))
                for i in range(num_claims):
                    for j in range(num_claims):
                        score = rouge.score(claims[i], claims[j])['rougeL'].fmeasure
                        dist_matrix[i][j] = 1.0 - score
                        dist_matrix[j][i] = dist_matrix[i][j]
                
                for th in distance_thresholds:
                    # In agglomerative clustering, threshold represents maximum distance
                    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=th)
                    try:
                        labels = clustering.fit_predict(dist_matrix)
                        diversity_scores[th] += len(set(labels))
                    except Exception:
                        diversity_scores[th] += num_claims # Fallback if algorithm fails on short lists
            else:
                for th in distance_thresholds:
                    diversity_scores[th] += num_claims

        n = len(graft_answers) if graft_answers else 1
        res = {
            "comprehensiveness_avg_claims": avg_claims / n,
            "citation_precision": citation_precision / n,
            "diversity_by_threshold": {str(th): val / n for th, val in diversity_scores.items()}
        }
        logger.info(f"Claim metrics computed: {res}")
        return res

    def standard_nlp_metrics(self, generated: List[str], references: List[str]) -> Dict[str, float]:
        """Computes ROUGE and BERTScore."""
        logger.info("Computing Standard NLP metrics (ROUGE, BERTScore)...")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        exact_matches = []
        
        for gen, ref in zip(generated, references):
            scores = scorer.score(ref, gen)
            for k in rouge_scores.keys():
                rouge_scores[k].append(scores[k].fmeasure)
            
            # Exact Match (for short answers)
            exact_matches.append(1.0 if gen.strip().lower() == ref.strip().lower() else 0.0)
            
        # BERTScore
        # Suppress verbosity during BERTScore to avoid clogging CLI
        P, R, F1 = bert_score.score(generated, references, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=False)
        
        return {
            "ROUGE-1": np.mean(rouge_scores['rouge1']),
            "ROUGE-2": np.mean(rouge_scores['rouge2']),
            "ROUGE-L": np.mean(rouge_scores['rougeL']),
            "BERTScore-F1": np.mean(F1.numpy()),
            "Exact_Match": np.mean(exact_matches)
        }

    def ragas_faithfulness_grounding(self, dataset_dict: Dict[str, List[Any]]) -> Dict[str, float]:
        """Evaluates Faithfulness and Grounding using RAGAS."""
        logger.info("Computing RAGAS Faithfulness & Grounding metrics...")
        
        # Format for RAGAS dataset
        from datasets import Dataset
        ragas_ds = Dataset.from_dict({
            "question": dataset_dict["questions"],
            "answer": dataset_dict["answers"],
            "contexts": dataset_dict["contexts"],
            "ground_truth": dataset_dict["ground_truths"]
        })
        
        try:
            from ragas.metrics import faithfulness, context_precision, answer_relevancy
            from ragas import evaluate
            from langchain_community.chat_models import ChatOllama
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.embeddings import OllamaEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            judge_llm = LangchainLLMWrapper(
                ChatOllama(
                    model="llama3.2:latest",
                    base_url="http://localhost:11434"
                )
            )
            
            ollama_embeddings = LangchainEmbeddingsWrapper(
                OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
            )
                
            # Note: Requires Ollama to be running
            result = evaluate(
                ragas_ds,
                metrics=[
                    faithfulness,
                    context_precision,
                    answer_relevancy
                ],
                llm=judge_llm,
                embeddings=ollama_embeddings,
                raise_exceptions=False
            )
            return {
                "Faithfulness": float(result.get("faithfulness") or 0.0),
                "Context_Precision": float(result.get("context_precision") or 0.0),
                "Answer_Relevancy": float(result.get("answer_relevancy") or 0.0)
            }
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}. Ensure GEMINI_API_KEY is set and Python version is \u22643.12.")
            return {"Faithfulness": 0.0, "Context_Precision": 0.0, "Answer_Relevancy": 0.0}

    def efficiency_metrics(self, system_name: str, queries: List[str], times_ms: List[float], token_costs: List[float], contexts: List[str]) -> Dict[str, Any]:
        """Calculates latency and contextual utilization metrics."""
        
        return {
            "system": system_name,
            "latency_mean_ms": np.mean(times_ms),
            "latency_p50_ms": np.percentile(times_ms, 50),
            "latency_p95_ms": np.percentile(times_ms, 95),
            "avg_token_cost": np.mean(token_costs),
            "context_utilization": np.mean([len(c.split())/4096 for c in contexts]) # Approximating 4096 max tokens
        }

    def fine_tuning_quality(self, train_losses: List[float], val_perplexities: List[float]) -> Dict[str, Any]:
        """Aggregates and formats fine-tuning specific metrics."""
        return {
            "train_loss_curve": train_losses,
            "val_perplexity_curve": val_perplexities,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_perplexity": val_perplexities[-1] if val_perplexities else None
        }

    def save_results(self, system_name: str, eval_results: Dict[str, Any]):
        """Accumulates evaluation results for each system tested."""
        if system_name not in self.results:
            self.results[system_name] = {}
        self.results[system_name].update(eval_results)
        
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Saved evaluation results for {system_name} to {metrics_file}")
