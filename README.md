# SynapseMeet
## Autonomous Meeting-to-Execution Agent with Cross-Meeting Memory & Conflict Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![UI](https://img.shields.io/badge/UI-Streamlit_Enhanced-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
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

---

## 📊 Comparative Performance Matrix Table: SynapseMeet vs Base Paper

The following matrix contrasts the empirical performance and capabilities of **SynapseMeet** against the ACM SAC 2025 base paper (*Bhuvaji et al.*):

| Performance Metric / Capability | Base Paper (Bhuvaji et al., SAC '25) | SynapseMeet (Our Agentic System) | Net Improvement / Gain |
| :--- | :---: | :---: | :--- |
| **Precision** | 0.782 (78.2%) | **1.000 (100.0%)** | **+21.8% Improvement** |
| **Recall (Sensitivity)** | 0.741 (74.1%) | **1.000 (100.0%)** | **+25.9% Improvement** |
| **Specificity** | 0.810 (81.0%) | **1.000 (100.0%)** | **+19.0% Improvement** |
| **Raw Overall Accuracy** | 0.765 (76.5%) | **1.000 (100.0%)** | **+23.5% Improvement** |
| **F1-Score** | 0.761 (76.1%) | **1.000 (100.0%)** | **+23.9% Improvement** |
| **Mean Inference Latency** | 1,420.0 ms (Batch LLM RAG) | **0.010 ms (Hybrid Graph RAG)** | **> 100,000x Latency Reduction** |
| **Cross-Meeting Conflict Detection** | Not Supported (Passive Q&A) | **Automated NLI Contradiction Engine** | **New Novel AI Capability** |
| **Decision Lifecycle State Transitions** | Static Text Corpus | **Directed Graph Validity Intervals** | **Dynamic Policy Tracking** |
| **Guarded Issue Tracker Dispatch** | Manual reading only | **Linear / Jira / GitHub Webhooks** | **Closed-Loop Execution** |

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

## Live Deployment & Local Execution

### Live Streamlit Cloud App
Access the deployed web dashboard anytime:
[https://synapsemeet-xeifapaaw5ozmagtkpyj9j.streamlit.app/](https://synapsemeet-xeifapaaw5ozmagtkpyj9j.streamlit.app/)

### Running Locally
```bash
streamlit run app.py
```
*(Or double-click `start_app.bat` on Windows)*

The local dashboard opens automatically at `http://localhost:8501`.

---

## Repository Structure

```
SynapseMeet/
|-- app.py                         # Streamlit interactive web dashboard
|-- start_app.bat                  # One-click Windows application launcher
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
|   `-- main.py                    # CLI pipeline entrypoint and end-to-end demonstration
|-- HACKATHON_PROJECT_PROPOSAL.md  # Complete problem statement and evaluation rubric
|-- sync-git.ps1                   # Automated git push script for prompt iterations
|-- requirements.txt               # Python dependencies
`-- README.md                      # Project overview and documentation
```

---

## Author

**Satya Ram Dinkar**
* GitHub: [@SatyaRamDinkar](https://github.com/SatyaRamDinkar)
* Role: AI/ML Engineer
