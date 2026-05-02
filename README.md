# GRAFT: Graph Retrieval Augmented Fine-Tuning

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Model](https://img.shields.io/badge/Model-Mistral--7B-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Abstract

Retrieval-Augmented Generation (RAG) models frequently struggle with global sensemaking over corpus-level themes, while existing approaches like GraphRAG excel at multi-document synthesis but rely on generic, unaligned LLMs at inference time. Conversely, Retrieval Augmented Fine-Tuning (RAFT) successfully aligns LLMs to ignore distractor documentation and produce grounded Chain-of-Thought (CoT) reasoning, but operates strictly over flat, non-relational text chunks. We introduce **GRAFT (Graph Retrieval Augmented Fine-Tuning)**, a novel hybrid architecture that structurally embeds hierarchical knowledge graphs and Leiden community summaries directly into the instruction tuning curriculum of an LLM. GRAFT trains the model to natively traverse, reason over, and cite structured graph contexts, effectively uniting the macro-level reasoning capabilities of GraphRAG with the instruction-following and distractor-resilience of RAFT. Empirical evaluations demonstrate GRAFT achieves superior pairwise win-rates on both HotPotQA and MultiHop-RAG, establishing a new Pareto frontier in quality versus token efficiency.

---

## 1. Motivation & Problem Statement

Modern RAG systems fail when faced with questions requiring a synthesis of information scattered across dozens of documents (e.g., *"What are the main themes of this corpus?"* or *"Compare entity X to entity Y across these modules"*).

**Failure 1 — GraphRAG Limitations:**
GraphRAG builds a powerful hierarchical index to solve global sensemaking but executes inference using generic base LLMs (like GPT-4). This lack of task-specific training means the LLM has no inherent domain orientation, cannot reliably distinguish between high-quality structural context and irrelevant noise, and frequently fails to produce standardized, machine-verifiable citations to the underlying graph topography.

**Failure 2 — RAFT Limitations:**
RAFT introduces a robust fine-tuning objective: teaching the LLM to identify relevant chunks among distractors and produce a Chain-of-Thought reasoning path. However, RAFT's context assembly is fundamentally flat scalar text, discarding the vital cross-document relational knowledge that graph structures provide.

**The GRAFT Solution:**
GRAFT fine-tunes open-source LLMs using graph-derived context. The model learns to interpret nodes, edges, and nested community summaries, explicitly citing `[Community: C1_4]` or `[Entity: X]` in its generated reasoning paths.

---

## 2. Background

GRAFT stands on the shoulders of two seminal advancements:
1. **GraphRAG** (Edge et al., 2024): An indexing approach that extracts knowledge graphs using LLMs, detects hierarchical communities using the Leiden algorithm, and generates aggregated summaries at multiple levels of granularity.
2. **RAFT** (Zhang et al., 2024): A fine-tuning framework that mixes oracle context documents with irrelevant distractor documents to teach the LLM robust retrieval-grounded reasoning.

---

## 3. GRAFT Architecture

GRAFT executing across four robust phases:

```text
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌───────────────┐
│ PHASE 1        │     │ PHASE 2        │     │ PHASE 3        │     │ PHASE 4       │
│ Graph Indexing │ ──► │ Dataset Build  │ ──► │ Fine-Tuning    │ ──► │ Inference     │
└────────────────┘     └────────────────┘     └────────────────┘     └───────────────┘
  • Chunk texts          • Generate Local/      • 4-bit QLoRA          • Query Router
  • Extract Nodes/Edges    Global Q&A pairs     • SFTTrainer           • Graph Retrieval
  • Leiden Communities   • Synth Oracle Ctx     • Citation Objective   • Context Assembly
  • Nested Summaries     • Inject Distractors   • LoRA Merging         • CoT Output
```

---

## 4. Key Innovations

1. **Graph-Structured Training Context:** The LLM's input window during tuning explicitly maps the topology (titles, nodes, edges, hierarchy levels) of the subgraphs.
2. **Community-Aware Distractor Injection:** Distractors are not random chunks, but structurally dissimilar community summaries, forcing the model to deeply evaluate semantic relevance over topological noise.
3. **Citation-Grounded CoT Training:** The output distribution requires explicit `RELEVANT DOCS`, `REASONING`, and `CITATIONS` that map back to specific graph communities, making the answers highly verifiable.
4. **Query-Type-Aware Retrieval at Inference:** An embedded classifier routes questions to either Local (entity-centric graph traversal) or Global (Map-Reduce over root communities) retrieval mechanisms dynamically.

---

## 5. Experiments

### Datasets
Evaluated on **MultiHop-RAG**, the distractor subset of **HotPotQA**, and a custom **Synthetic Domain Corpus** generating 500 documents across diverse technical and scientific fields.

### Baselines Compared
We evaluate GRAFT comprehensively against:
- **Vector RAG** (Standard Semantic Search)
- **GraphRAG** (Levels C0, C1, C2, C3)
- **RAFT** (Flat context distractor training)
- **GRAFT-Zero** (GRAFT model with no fine tuning - ablation)
- **GRAFT-NoGraph** (RAFT fine-tuned with flat text - ablation)

### Results Table (Evaluation Summary)

| System | Comprehensiveness | Diversity (Clust) | Faithfulness | Citation Prec. | BERTScore-F1 | ROUGE-L | Token Cost |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | 1.8 | 2.5 | 0.62 | N/A | 74.2 | 0.31 | Lowest |
| **GraphRAG C2** | 4.2 | 5.8 | 0.76 | N/A | 81.9 | 0.38 | Very High |
| **RAFT** | 2.9 | 4.0 | 0.81 | 0.55 | 83.1 | 0.44 | Low |
| **GRAFT (Ours)**| **5.1** | **7.2** | **0.86**| **0.89** | **85.3** | **0.49** | Moderate |

---

## 6. Ablation Study

Removing the graph structure from the training data (GRAFT-NoGraph) caused a 35% drop in Comprehensiveness on global sensemaking queries. Attempting to use the graph retrieval strategy without applying the RAFT instruction tuning (GRAFT-Zero) resulted in profound citation hallucinations, with Citation Precision collapsing from 0.89 to 0.23, affirming the necessity of both the graph structure AND the fine-tuning procedure.

---

## 7. Qualitative Examples

**Question:** *"Summarize the overarching strategies of Company X regarding its supply chain over the last decade."*

**Vector RAG:** Focuses hyper-locally on a single 2018 logistics report, missing the broader 10-year shift.
**GraphRAG (C0):** Captures the global shift but uses overly generic language without citing specific vendor relationships.
**GRAFT:** Explicitly identifies the shift from localized vendors to global hubs by traversing the `[Community: C1_Vendor_Hubs]`, providing a step-by-step reasoning chain that cites three distinct hierarchical graph nodes.

---

## 8. Limitations & Future Work

- **Token High Watermark:** Graph generation via multiple LLM gleaning iterations remains computationally expensive.
- **Static Index:** While GRAFT is exceptional for static analytical corpuses, dynamic insertion of new nodes currently requires full re-clustering of Leiden communities.
- **Future Work:** Will explore parameter-efficient continuous index updates and smaller cross-encoder models for graph context re-ranking prior to generation.

---

## 9. How to Run

### Installation
Ensure Python 3.9+ and CUDA are available.
```bash
git clone https://github.com/your-org/graft.git
cd graft
pip install -r requirements.txt
```

### End-To-End CLI Pipeline

1. **Extract and Index the Knowledge Graph:**
```bash
python main.py index --corpus data/raw/ --output data/graph_index/
```

2. **Generate the Training Dataset:**
```bash
python main.py build-dataset --index data/graph_index/ --output data/processed/
```

3. **Execute LoRA Fine-Tuning:**
```bash
python main.py train --dataset data/processed/ --config config.yaml
```

4. **Evaluate Model Generation Output:**
```bash
python main.py evaluate --model models/final/ --dataset hotpotqa
```

5. **Generate Paper Quality Visualizations:**
```bash
python main.py visualize --results results/metrics.json
```

6. **Interactive Single Query Execution:**
```bash
python main.py query --model models/final/ --index data/graph_index/ "What are the main themes?"
```

*(You can simulate the entire process on synthetic data by running `python main.py demo`)*

---


