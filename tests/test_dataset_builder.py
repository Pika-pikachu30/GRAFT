import pytest
from unittest.mock import MagicPatch
import networkx as nx
from src.raft_dataset_builder import RAFTDatasetBuilder, Question, ContextDocument, CoTAnswer
from src.graph_indexer import CommunitySummary

@pytest.fixture
def mock_builder():
    return RAFTDatasetBuilder(llm_wrapper=None)

def test_dataset_distractor_injection(mock_builder):
    oracle_context = [
        ContextDocument(id="doc_C0_1", text="Oracle summary", is_oracle=True, source="C0_1")
    ]
    
    all_summaries = {
        "C0_1": CommunitySummary(level=0, summary="Oracle summary", title="Test", rating=5.0, nodes=[], tokens=10),
        "C0_2": CommunitySummary(level=0, summary="Distract 1", title="D1", rating=1.0, nodes=[], tokens=10),
        "C0_3": CommunitySummary(level=0, summary="Distract 2", title="D2", rating=1.0, nodes=[], tokens=10),
        "C0_4": CommunitySummary(level=0, summary="Distract 3", title="D3", rating=1.0, nodes=[], tokens=10)
    }
    
    aug_context = mock_builder.inject_distractor_documents(oracle_context, all_summaries, n_distractors=2)
    
    assert len(aug_context) == 3 # 1 oracle + 2 distractors
    assert sum(1 for d in aug_context if d.is_oracle) == 1
    assert "C0_1" in [d.source for d in aug_context]
    
def test_build_training_sample(mock_builder):
    q = Question(id="q1", text="What is X?", type="local", source_community="C1")
    aug = [
        ContextDocument(id="doc_C1", text="X is Y", is_oracle=True, source="C1"),
        ContextDocument(id="doc_D1", text="Distractor", is_oracle=False, source="D1")
    ]
    ans = CoTAnswer(
        relevant_docs=["C1"],
        reasoning="I found X in doc C1.",
        final_answer="Y",
        citations=["[C1]"]
    )
    
    sample = mock_builder.build_training_sample(q, aug, ans)
    
    assert "What is X?" in sample.instruction
    assert "X is Y" in sample.input
    assert "Distractor" in sample.input
    assert "Let's think step by step." in sample.output
    assert "I found X in doc C1." in sample.output
    assert "[C1]" in sample.output
