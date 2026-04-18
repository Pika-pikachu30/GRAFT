import pytest
import os
import networkx as nx
from src.graph_indexer import GraphIndexer, Chunk, CommunitySummary

@pytest.fixture
def mock_indexer():
    # Pass a dummy llm wrapper if we needed to, but we can test logic without real network calls
    return GraphIndexer(llm_wrapper=None)

def test_chunk_documents(mock_indexer):
    texts = [
        "This is a test document that should be chunked into smaller pieces if the size is very small compared to this string."
    ]
    chunks = mock_indexer.chunk_documents(texts, chunk_size=5, overlap=2)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].source_doc == "doc_0"

def test_graph_parsing_logic(mock_indexer):
    mock_response = (
        "ENTITY: Global Corp | Company | A massive tech conglom\n"
        "ENTITY: Alice | Person | CEO of Global Corp\n"
        "RELATION: Alice | Global Corp | manages | 9.5"
    )
    mock_indexer._parse_and_update_graph(mock_response)
    
    assert mock_indexer.graph.number_of_nodes() == 2
    assert "Global Corp" in mock_indexer.graph
    assert "Alice" in mock_indexer.graph
    assert mock_indexer.graph.nodes["Alice"].get("type") == "Person"
    
    assert mock_indexer.graph.number_of_edges() == 1
    assert mock_indexer.graph.has_edge("Alice", "Global Corp")
    assert mock_indexer.graph["Alice"]["Global Corp"]["weight"] == 9.5

def test_community_detection_hierarchy(mock_indexer):
    # Manually create a graph with two distinct cliques
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4)])
    
    hierarchy = mock_indexer.detect_communities(G)
    
    # Needs to return a 4 level (0-3) dict structure
    assert isinstance(hierarchy, dict)
    assert 0 in hierarchy
    assert 3 in hierarchy
    
    for level, comms in hierarchy.items():
        assert isinstance(comms, dict)
        if level == 3: # Leaf level should capture all nodes in some community
            all_nodes = [n for nodes in comms.values() for n in nodes]
            assert set(all_nodes) == set(G.nodes())

def test_save_and_load_index(tmp_path, mock_indexer):
    G = nx.Graph()
    G.add_edge("A", "B", weight=2.0)
    mock_indexer.graph = G
    
    mock_indexer.community_summaries = {
        "C0_1": CommunitySummary(level=0, summary="Test summary", title="Test", rating=5.0, nodes=["A", "B"], tokens=10)
    }
    
    index_dir = tmp_path / "index_test"
    mock_indexer.save_index(str(index_dir))
    
    assert (index_dir / "knowledge_graph.graphml").exists()
    assert (index_dir / "community_summaries.json").exists()
    
    new_indexer = GraphIndexer()
    new_indexer.load_index(str(index_dir))
    
    assert new_indexer.graph.has_edge("A", "B")
    assert "C0_1" in new_indexer.community_summaries
    assert new_indexer.community_summaries["C0_1"].title == "Test"
