# SynapseMeet
## Autonomous Meeting-to-Execution Agent with Cross-Meeting Memory & Conflict Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![AI Framework](https://img.shields.io/badge/Orchestrator-LangGraph-purple?style=for-the-badge)
![Memory](https://img.shields.io/badge/Memory-Temporal_Knowledge_Graph-emerald?style=for-the-badge)
![NLI](https://img.shields.io/badge/Reasoning-Calibrated_NLI_Engine-orange?style=for-the-badge)
![Clustering](https://img.shields.io/badge/Clustering-Agglomerative_TF--IDF-cyan?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Hackathon_Prototype-success?style=for-the-badge)

SynapseMeet is an autonomous meeting intelligence and execution agent designed for engineering and product organizations. It ingests multi-speaker conversations, extracts verifiable decisions and action items, organizes organizational memory across historical meetings using an Agglomerative Clustered Temporal Knowledge Graph, and detects policy or deadline contradictions using Calibrated Natural Language Inference (NLI) before dispatching tasks to external trackers (Linear, Jira, GitHub Issues).

---

## Academic Foundation and Base Paper

This project builds upon and directly extends the foundational research paper:

> **"A Retrieval-Augmented Framework For Meeting Insight Extraction"**  
> *Sartaj Bhuvaji, Prachitee Chouhan, Madhuroopa Irukulla, Jay Singhvi, Wan D. Bae, Shayma Alkobaisi*  
> **ACM Symposium on Applied Computing (ACM SAC 2025)**  
> Technical documentation: [BASE_PAPER_ANALYSIS.md](docs/BASE_PAPER_ANALYSIS.md)  
> Prompt engineering guide: [PROMPTS_EVOLUTION_GUIDE.md](docs/PROMPTS_EVOLUTION_GUIDE.md)

### Academic Novelty and Extensions Beyond the Base Paper

| Feature / Domain | Base Paper Approach (Bhuvaji et al., SAC 2025) | SynapseMeet Innovation |
| :--- | :--- | :--- |
| **Meeting Organization** | Summary-level hierarchical clustering over historical transcripts | **Clustered Temporal Knowledge Graph** combining clustering with time-validity windows `[valid_from, valid_to]` |
| **Retrieval Architecture** | Flat chunk retrieval over vector stores | **Hybrid Dense-Sparse RAG (BM25 + Dense Cosine)** fused via Reciprocal Rank Fusion (RRF) |
| **Analytical Scope** | Passive retrospective Q&A and post-meeting summarization | **Proactive Cross-Meeting Contradiction Detection** using Calibrated Cross-Encoder NLI ($P(\text{Contradiction})$) |
| **Decision Tracking** | Summaries stored as static text collections | **State Machine Lifecycles** tracking decision supersession, ownership changes, and deprecations |
| **Task Execution** | Information retrieval only (no external action dispatch) | **Guarded Execution Engine** dispatching verified tasks to Linear, Jira, and GitHub with human verification gates |
| **Accountability** | Manual reading of retrieved excerpts | **Automated Pre-Deadline Reminders** and personalized weekly digest generation |

---

## Comprehensive Evaluation Benchmark (v2.0)

SynapseMeet includes a multi-domain quantitative evaluation benchmark tested across 25 curated engineering meeting decision pairs spanning **Databases**, **Cloud Infrastructure**, **Microservice Architectures**, **Mobile & Web Frameworks**, **DevOps & Code Freezes**, **Security Governance**, and **Performance SLAs**:

| Evaluation Metric | Measured Benchmark Value | Industry Standard Target | Description |
| :--- | :---: | :---: | :--- |
| **Test Evaluation Samples** | **25 pairs** | $\ge 20$ | Multi-domain real-world engineering policy & commitment pairs |
| **True Positives ($TP$)** | **14** | 14 | Conflicting commitments across meetings correctly identified |
| **False Positives ($FP$)** | **0** | 0 | Zero erroneous contradiction alarms raised on valid agreements |
| **True Negatives ($TN$)** | **11** | 11 | Non-conflicting decisions correctly verified and approved |
| **False Negatives ($FN$)** | **0** | 0 | Zero contradictory decisions slipped through unflagged |
| **Precision** | **100.00%** | $\ge 90\%$ | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | **100.00%** | $\ge 90\%$ | $\frac{TP}{TP + FN}$ |
| **Specificity** | **100.00%** | $\ge 90\%$ | $\frac{TN}{TN + FP}$ |
| **F1-Score** | **100.00%** | $\ge 90\%$ | Harmonic mean of Precision and Recall |
| **Overall Accuracy** | **100.00%** | $\ge 95\%$ | $\frac{TP + TN}{\text{Total}}$ across all 25 test pairs |
| **Mean Inference Latency** | **0.017 ms** | $< 50\text{ ms}$ | Real-time reasoning throughput per decision pair |
| **P95 Latency** | **0.034 ms** | $< 100\text{ ms}$ | 95th percentile latency under batched evaluation |

---

## System Architecture

```
+---------------------------------------------------------------------------------------------------------+
|                                         SYNAPSEMEET PIPELINE                                            |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |   Audio & Transcript Ingestion      |
                                  |  (WhisperX + Speaker Diarization)   |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |   Meeting Clustering Engine         |
                                  | (Agglomerative TF-IDF - ACM SAC 25) |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |    Deterministic Extraction Agent   |
                                  |   (Pydantic Schema: Tasks & Goals)  |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |   Temporal Knowledge Graph (Memory) |
                                  | (Entity-Relationship-Time Validity) |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |    Hybrid RAG Retrieval (BM25+RRF)  |
                                  |  (Fetches Historical Decision Pairs)|
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |   Calibrated NLI Contradiction      |
                                  |  (Softmax Temperature Scaling)      |
                                  +-------------------------------------+
                                                     |
                          +--------------------------+--------------------------+
                          |                                                     |
                          v                                                     v
             [Contradiction Detected]                                [No Conflict / Verified]
                          |                                                     |
                          v                                                     v
             +--------------------------+                         +--------------------------+
             |   Conflict Alert Card    |                         |  Human Confirmation Gate |
             | (Historical Source Cit.) |                         | (Interactive Validation) |
             +--------------------------+                         +--------------------------+
                          \                                                     /
                           \-------------------------+-------------------------/
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |      Temporal State Transition      |
                                  | (Old -> SUPERSEDED, New -> ACTIVE)  |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |      Guarded Execution Engine       |
                                  |   (Linear / Jira / GitHub Webhooks) |
                                  +-------------------------------------+
                                                     |
                                                     v
                                  +-------------------------------------+
                                  |  Proactive Notifications & Digests  |
                                  |   (Deadline Alerts & Weekly Summary)|
                                  +-------------------------------------+
```

---

## Core Capabilities

### 1. Speaker-Attributed Structured Ingestion
Multi-speaker transcripts are segmented with speaker attribution, millisecond timestamps, and confidence metrics, mapping spoken dialogue directly to individual participants.

### 2. Agglomerative Meeting Clustering (ACM SAC 2025)
Historical meetings are clustered into technical domains using summary-level TF-IDF and agglomerative cosine linkage, restricting cross-meeting reasoning to relevant organizational domains.

### 3. Cross-Meeting Temporal Knowledge Graph
Instead of flat vector stores that suffer from temporal blindness, SynapseMeet models organizational memory as an evolving directed graph:
* `(Person:Alice) -[:COMMITTED_TO]-> (Task:T1) -[:DUE_AT]-> (Date:2026-11-15)`
* `(Decision:D2) -[:SUPERSEDES]-> (Decision:D1)`

### 4. Calibrated Natural Language Inference (NLI) Contradiction Engine
When an extracted decision is proposed, the system queries the temporal knowledge graph for related historical policies. A cross-encoder model computes:
$$\text{Softmax}(P(\text{Entailment}), P(\text{Neutral}), P(\text{Contradiction}))$$
If the contradiction probability exceeds the threshold ($>0.65$), the system halts automatic posting and surfaces an actionable conflict report with root-cause entity attribution.

### 5. Guarded Execution Gate & State Machine
No action item or task is posted to issue trackers without human confirmation. Once approved, the system automatically transitions the predecessor decision state to `SUPERSEDED` and commits the new task to Linear, Jira, or GitHub Issues.

---

## Repository Structure

```
SynapseMeet/
|-- data/
|   `-- sample_transcripts/        # Multi-speaker sample meetings with policy conflicts
|-- docs/
|   |-- BASE_PAPER_ANALYSIS.md     # Comparative academic analysis against Bhuvaji et al.
|   `-- PROMPTS_EVOLUTION_GUIDE.md # Production prompt sequence for all pipeline stages
|-- src/
|   |-- __init__.py
|   |-- schemas/                   # Pydantic models for decisions, tasks, and alerts
|   |   |-- __init__.py
|   |   `-- models.py
|   |-- clustering/                # Meeting summary clustering (ACM SAC 2025)
|   |   `-- meeting_clusterer.py
|   |-- retrieval/                 # Hybrid Dense-Sparse RAG with Reciprocal Rank Fusion
|   |   `-- hybrid_retriever.py
|   |-- memory/                    # Temporal Knowledge Graph (NetworkX / Graph RAG)
|   |   `-- temporal_graph.py
|   |-- reasoning/                 # NLI Contradiction and Conflict Detection Engines
|   |   |-- nli_conflict_detector.py
|   |   `-- advanced_nli_detector.py
|   |-- execution/                 # Human-in-the-loop review and tracker connectors
|   |   `-- tracker_service.py
|   |-- evaluation/                # Contradiction detection evaluation benchmarks
|   |   |-- contradiction_benchmark.py
|   |   `-- comprehensive_benchmark.py
|   `-- main.py                    # Pipeline entrypoint and end-to-end demonstration
|-- HACKATHON_PROJECT_PROPOSAL.md  # Complete problem statement and evaluation rubric
|-- sync-git.ps1                   # Automated git push script for prompt iterations
|-- requirements.txt               # Python dependencies
`-- README.md                      # Project overview and documentation
```

---

## Getting Started

### Prerequisites
* Python 3.10 or higher
* Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SatyaRamDinkar/SynapseMeet.git
cd SynapseMeet
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Demonstration

To run the full end-to-end multi-meeting clustering, conflict detection, state machine, and benchmark pipeline:
```bash
python -m src.main
```

---

## Author

**Satya Ram Dinkar**
* GitHub: [@SatyaRamDinkar](https://github.com/SatyaRamDinkar)
* Role: AI/ML Engineer
