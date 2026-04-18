import csv
import json
import os
import uuid
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import igraph as ig
import leidenalg
from utils import setup_logger, LLMWrapper

logger = setup_logger(__name__)

@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    id: str
    text: str
    source_doc: str
    token_count: int

@dataclass
class CommunitySummary:
    """Represents a summarized community in the Knowledge Graph."""
    level: int
    summary: str
    title: str
    rating: float
    nodes: List[str]
    tokens: int
    sub_communities: List[str] = field(default_factory=list)

class GraphIndexer:
    """
    Handles Phase 1 of GRAFT: Indexing documents into a hierarchical knowledge graph.
    """
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        self.llm = llm_wrapper or LLMWrapper()
        self.graph = nx.Graph()
        self.hierarchical_communities: Dict[int, Dict[str, List[str]]] = {}
        self.community_summaries: Dict[str, CommunitySummary] = {}

    def load_documents(self, path: str) -> List[str]:
        """Loads documents from a given path (file or directory)."""
        texts = []
        if os.path.isfile(path):
            texts.extend(self._read_file(path))
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                if os.path.isfile(filepath):
                    texts.extend(self._read_file(filepath))
        else:
            raise ValueError(f"Invalid path: {path}")
            
        logger.info(f"Loaded {len(texts)} documents from {path}")
        return texts

    def _read_file(self, filepath: str) -> List[str]:
        """Helper to read individual files based on extension."""
        texts = []
        ext = filepath.split(".")[-1].lower()
        try:
            if ext == "txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif ext == "jsonl":
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        texts.append(data.get("text", line))
            elif ext == "csv":
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        texts.append(" ".join(row))
            elif ext == "pdf":
                # Fallback simple text extraction or placeholder 
                # (For production, PyPDF2 or pdfplumber is recommended)
                logger.warning("PDF reading assumes pre-extracted text or requires PyPDF2. Reading as plain text fallback.")
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                     texts.append(f.read())
            else:
                logger.warning(f"Unsupported file format: {ext}")
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            
        return [text.strip() for text in texts if text.strip()]

    def chunk_documents(self, texts: List[str], chunk_size: int = 600, overlap: int = 100) -> List[Chunk]:
        """Splits documents into overlapping chunks."""
        chunks = []
        for doc_idx, text in enumerate(texts):
            # Simple space-based tokenizer approximation for fast chunking
            words = text.split()
            if not words:
                continue
                
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source_doc=f"doc_{doc_idx}",
                    token_count=len(chunk_words) # Approximate token count
                ))
                
                if end == len(words):
                    break
                start += (chunk_size - overlap)
                
        logger.info(f"Created {len(chunks)} chunks from {len(texts)} documents.")
        return chunks

    def extract_entities_and_relations(self, chunks: List[Chunk], max_glean_iterations: int = 3) -> nx.Graph:
        """Uses LLM to extract entity-relation graphs from text chunks using self-reflection gleaning."""
        for chunk in chunks:
            # Gleaning protocol
            extracted_data = ""
            for iteration in range(max_glean_iterations + 1):
                if iteration == 0:
                    prompt = self._build_extraction_prompt(chunk.text)
                else:
                    prompt = self._build_gleaning_prompt(chunk.text, extracted_data)
                    
                response = self.llm.generate(prompt=prompt)
                
                # Check for stopping condition in gleaning
                if iteration > 0 and "NO_MISSED_ENTITIES" in response:
                    break
                    
                extracted_data += f"\n{response}"
                self._parse_and_update_graph(response)

        logger.info(f"Extracted Knowledge Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def _build_extraction_prompt(self, text: str) -> str:
        return (
            "Extract all significant entities and the relationships between them from the following text.\n"
            "Format the output strictly as follows:\n"
            "ENTITY: [entity_name] | [entity_type] | [entity_description]\n"
            "RELATION: [source_entity_name] | [target_entity_name] | [relation_description] | [strength_1_to_10]\n\n"
            f"Text:\n{text}\n"
        )

    def _build_gleaning_prompt(self, text: str, previous_extraction: str) -> str:
        return (
            "Review the original text and the entities/relations already extracted.\n"
            "Are there any significant entities or relations missed? If completely thorough, output 'NO_MISSED_ENTITIES'.\n"
            "If there are missed elements, list them in the exact same format:\n"
            "ENTITY: ...\nRELATION: ...\n\n"
            f"Original Text:\n{text}\n\nPrevious Extracted Data:\n{previous_extraction}"
        )

    def _parse_and_update_graph(self, response: str):
        """Parses the LLM output and updates the NetworkX graph."""
        pending_relations = []
        for line in response.split("\n"):
            line = line.strip().replace("**", "").replace("*", "")
            if line.startswith("ENTITY:"):
                parts = [p.strip() for p in line[7:].split("|")]
                if len(parts) >= 3:
                    name, e_type, desc = parts[0], parts[1], parts[2]
                    if not self.graph.has_node(name):
                        self.graph.add_node(name, type=e_type, description=desc)
                    else:
                        # Append description if exists
                        curr_desc = self.graph.nodes[name].get('description', '')
                        self.graph.nodes[name]['description'] = f"{curr_desc} {desc}".strip()
                        
            elif line.startswith("RELATION:"):
                parts = [p.strip() for p in line[9:].split("|")]
                if len(parts) >= 4:
                    pending_relations.append(parts)

        for parts in pending_relations:
            source, target, r_desc = parts[0], parts[1], parts[2]
            try:
                strength = float(parts[3])
            except ValueError:
                strength = 1.0
                
            for n in (source, target):
                if not self.graph.has_node(n):
                    self.graph.add_node(n, type="Unknown", description="")
                    
            if self.graph.has_edge(source, target):
                self.graph[source][target]['weight'] += strength
                self.graph[source][target]['description'] += f"; {r_desc}"
            else:
                self.graph.add_edge(source, target, weight=strength, description=r_desc)

    def detect_communities(self, graph: nx.Graph) -> Dict[int, Dict[str, List[str]]]:
        """Runs Leiden community detection to build a 4-level hierarchy (C0 to C3)."""
        if graph.number_of_nodes() == 0:
            return {}

        # Convert networkx to igraph for Leiden algorithm
        ig_graph = ig.Graph.TupleList(graph.edges(data=True), weights=True)
        # Ensure all nodes are mapped (isolated nodes)
        for node in graph.nodes():
            if node not in ig_graph.vs['name']:
                ig_graph.add_vertex(name=node)
                
        # Run recursive Leiden detection to build hierarchy
        # C3 is leaf (most granular), C0 is root (most global)
        hierarchy = {}
        current_partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
        
        # Simplified simulated hierarchy building for 4 levels
        for level in range(4): # 0, 1, 2, 3
            level_dict = {}
            has_name = 'name' in ig_graph.vs.attributes()
            for comm_idx, members in enumerate(current_partition):
                comm_id = f"C{level}_{comm_idx}"
                if has_name:
                    node_names = [ig_graph.vs[m]['name'] for m in members]
                else:
                    node_names = [f"Node_{m}" for m in members]
                level_dict[comm_id] = node_names
            
            hierarchy[level] = level_dict
            
            # Aggregate graph for next level up (simplification trick for simulation)
            ig_graph.contract_vertices(current_partition.membership)
            ig_graph.simplify(combine_edges=dict(weight="sum"))
            
            # Reassign names to contracted communities so the next level has valid names
            ig_graph.vs['name'] = [f"C{level}_{i}" for i in range(len(current_partition))]
            
            current_partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

        self.hierarchical_communities = hierarchy
        logger.info(f"Detected 4-level community hierarchy with {len(hierarchy[0])} root communities.")
        return hierarchy

    def summarize_communities(self, hierarchy: Dict[int, Dict[str, List[str]]], graph: nx.Graph) -> Dict[str, CommunitySummary]:
        """Summarizes communities bottom-up using LLM."""
        summaries = {}
        
        # Process bottom-up: C3 (leaves) -> C0 (root)
        for level in sorted(hierarchy.keys(), reverse=True):
            for comm_id, nodes in hierarchy[level].items():
                if level == 3:
                    # Leaf community: summarize raw nodes and edges
                    # Fix: Flatten the comma-separated contracted node names back
                    actual_nodes = []
                    for n in nodes:
                        actual_nodes.extend(n.split(','))
                    
                    subgraph = graph.subgraph(actual_nodes)
                    context_text = self._build_subgraph_context(subgraph)
                    prompt = (
                        f"Summarize the following graph community into a comprehensive report. "
                        f"Include a title, key entities, and main relationships and claims.\n\n{context_text}"
                    )
                else:
                    # Higher levels: summarize sub-communities
                    # Find sub-communities from the level below (level+1)
                    sub_comms = [c for c, c_nodes in hierarchy[level + 1].items() if set(c_nodes).issubset(set(nodes))]
                    sub_comm_texts = [summaries[c].summary for c in sub_comms if c in summaries]
                    context_text = "\n\n".join(sub_comm_texts)
                    prompt = (
                        f"Synthesize the following sub-community summaries into a higher-level global summary. "
                        f"Produce a title, overall rating of importance (1.0-10.0), and a cohesive narrative.\n\n{context_text}"
                    )
                
                # Assume empty if context is empty
                if not context_text.strip():
                    continue
                    
                response = self.llm.generate(prompt=prompt)
                
                # Parse title and rating manually or via prompt strictness
                title_match = re.search(r'Title:\s*(.*)', response, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else f"{comm_id} Summary"
                
                rating_match = re.search(r'Rating:\s*([0-9.]+)', response, re.IGNORECASE)
                rating = float(rating_match.group(1)) if rating_match else 5.0
                
                summary_obj = CommunitySummary(
                    level=level,
                    summary=response,
                    title=title,
                    rating=rating,
                    nodes=nodes,
                    tokens=len(response.split())
                )
                summaries[comm_id] = summary_obj
                
        self.community_summaries = summaries
        
        if len(summaries) == 0:
            # Use node names as fallback summaries
            for i, node in enumerate(graph.nodes()):
                summary_obj = CommunitySummary(
                    level=3,
                    summary=f"Entity: {node}",
                    title=f"Entity: {node}",
                    rating=5.0,
                    nodes=[node],
                    tokens=3
                )
                summaries[f"Fallback_{i}"] = summary_obj

        logger.info(f"Generated {len(summaries)} community summaries.")
        return summaries

    def _build_subgraph_context(self, subgraph: nx.Graph) -> str:
        """Serializes a subgraph into text for the LLM."""
        lines = []
        for node, data in subgraph.nodes(data=True):
            lines.append(f"Entity: {node} ({data.get('type', 'Unknown')}) - {data.get('description', '')}")
        for u, v, data in subgraph.edges(data=True):
            lines.append(f"Relation: {u} -> {v} [Weight: {data.get('weight', 1.0)}] - {data.get('description', '')}")
        return "\n".join(lines)

    def save_index(self, path: str):
        """Saves the NetworkX graph and community summaries."""
        os.makedirs(path, exist_ok=True)
        # Save graph
        graph_path = os.path.join(path, "knowledge_graph.graphml")
        nx.write_graphml(self.graph, graph_path)
        
        # Save summaries
        summaries_dict = {
            c_id: {
                "level": c.level,
                "summary": c.summary,
                "title": c.title,
                "rating": c.rating,
                "nodes": c.nodes,
                "tokens": c.tokens
            } for c_id, c in self.community_summaries.items()
        }
        with open(os.path.join(path, "community_summaries.json"), "w", encoding="utf-8") as f:
            json.dump(summaries_dict, f, indent=4)
            
        logger.info(f"Saved Graph index to {path}")

    def load_index(self, path: str):
        """Loads the index from disk."""
        graph_path = os.path.join(path, "knowledge_graph.graphml")
        if os.path.exists(graph_path):
            self.graph = nx.read_graphml(graph_path)
            
        summaries_path = os.path.join(path, "community_summaries.json")
        if os.path.exists(summaries_path):
            with open(summaries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.community_summaries = {
                    k: CommunitySummary(**v) for k, v in data.items()
                }
        logger.info(f"Loaded index from {path}")
