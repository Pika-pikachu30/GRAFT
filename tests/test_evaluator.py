import pytest
import numpy as np
from src.evaluator import Evaluator
from src.graft_inference import GRAFTAnswer

@pytest.fixture
def evaluator():
    return Evaluator(llm_wrapper=None)

def test_claim_metrics(evaluator):
    """Test claim parsing logic (Comprehensiveness & Citation Precision) without breaking on ML imports."""
    answers = [
        GRAFTAnswer(
            relevant_docs=["C1", "C2"],
            reasoning_chain="...",
            final_answer="This is a huge claim about science. And another fact that is very long.",
            citations=["[C1]", "[C3]"]
        )
    ]
    
    # Needs 2 claims since we simulated sentence-splitting
    res = evaluator.claim_metrics(answers, [], distance_thresholds=[0.5])
    
    assert res["comprehensiveness_avg_claims"] == 2.0
    assert res["citation_precision"] == 0.5  # Only C1 is relevant, C3 is an hallucination or miss
    assert "0.5" in res["diversity_by_threshold"]

def test_nlp_metrics(evaluator):
    gen = ["The quick brown fox jumps over the lazy dog"]
    ref = ["A fast brown fox jumps over the lazy dog"]
    
    res = evaluator.standard_nlp_metrics(gen, ref)
    
    assert res["ROUGE-1"] > 0.0
    assert res["ROUGE-L"] > 0.0
    assert res["Exact_Match"] == 0.0 # Differs slightly
    assert res["BERTScore-F1"] > 0.8 # Similarity is high

def test_efficiency_metrics(evaluator):
    res = evaluator.efficiency_metrics(
        "GRAFT",
        queries=["Q1", "Q2"],
        times_ms=[1500, 2500],
        token_costs=[4000, 6000],
        contexts=["Small test text 1 2 3", "Another larger test string 1 2 3 4 5 6"]
    )
    
    assert res["latency_mean_ms"] == 2000.0
    assert res["avg_token_cost"] == 5000.0
    assert res["context_utilization"] < 0.1 # Very few words out of 4096 tokens
