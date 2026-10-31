# SynapseMeet
## Autonomous Meeting-to-Execution Agent with Cross-Meeting Memory & Conflict Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![AI Framework](https://img.shields.io/badge/Orchestrator-LangGraph-purple?style=for-the-badge)
![Memory](https://img.shields.io/badge/Memory-Temporal_Knowledge_Graph-emerald?style=for-the-badge)
![NLI](https://img.shields.io/badge/Reasoning-NLI_Contradiction_Engine-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Hackathon_Prototype-success?style=for-the-badge)

SynapseMeet is an autonomous meeting intelligence and execution agent designed for engineering and product teams. It ingests multi-speaker conversations, extracts verifiable decisions and action items, tracks organizational memory across multiple meetings using a Temporal Knowledge Graph, and detects policy or deadline contradictions using Natural Language Inference (NLI) before dispatching tasks to external trackers (Linear, Jira, GitHub Issues).

---

## Academic Foundation and Base Paper

This project builds upon and extends the foundational research paper:

> **"A Retrieval-Augmented Framework For Meeting Insight Extraction"**  
> *Sartaj Bhuvaji, Prachitee Chouhan, Madhuroopa Irukulla, Jay Singhvi, Wan D. Bae, Shayma Alkobaisi*  
> **ACM Symposium on Applied Computing (ACM SAC 2025)**  
> Technical documentation: [BASE_PAPER_ANALYSIS.md](docs/BASE_PAPER_ANALYSIS.md)

### Academic Novelty and Extensions Beyond the Base Paper

| Feature / Domain | Base Paper Approach (Bhuvaji et al., SAC 2025) | SynapseMeet Innovation |
| :--- | :--- | :--- |
| **Retrieval Architecture** | Flat chunk retrieval and summary clustering over past transcripts | **Temporal Knowledge Graph (Graph RAG)** with temporal validity windows `[valid_from, valid_to]` |
| **Analytical Scope** | Passive retrospective Q&A and post-meeting summarization | **Proactive Cross-Meeting Contradiction Detection** using Cross-Encoder NLI ($P(\text{Contradiction})$) |
| **Decision Tracking** | Summaries stored in static vector collections | **State Machine Lifecycles** tracking decision supersession, ownership changes, and deprecations |
| **Task Execution** | Information retrieval only (no external action dispatch) | **Guarded Execution Engine** dispatching verified tasks to Linear, Jira, and GitHub with human verification gates |
| **Accountability** | Manual reading of retrieved excerpts | **Automated Deadline Reminders** and personalized weekly digest generation |

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
                                  |     NLI Contradiction Engine        |
                                  |  (Cross-Encoder Entailment Check)   |
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

### 2. Cross-Meeting Temporal Knowledge Graph
Instead of flat vector stores that suffer from temporal blindness, SynapseMeet models organizational memory as an evolving directed graph:
* `(Person:Alice) -[:COMMITTED_TO]-> (Task:T1) -[:DUE_AT]-> (Date:2026-11-15)`
* `(Decision:D2) -[:SUPERSEDES]-> (Decision:D1)`

### 3. Natural Language Inference (NLI) Contradiction Engine
When an extracted decision is proposed, the system queries the temporal knowledge graph for related historical policies. A cross-encoder model computes:
$$\text{Softmax}(P(\text{Entailment}), P(\text{Neutral}), P(\text{Contradiction}))$$
If the contradiction probability exceeds the threshold ($>0.70$), the system halts automatic posting and surfaces an actionable conflict report.

### 4. Guarded Execution Gate
No action item or task is posted to issue trackers without human confirmation. The system displays exact evidence from previous meetings, allowing operators to override or maintain historical policies.

---

## Repository Structure

```
SynapseMeet/
|-- data/
|   `-- sample_transcripts/        # Multi-speaker sample meetings with policy conflicts
|-- docs/
|   `-- BASE_PAPER_ANALYSIS.md     # Comparative academic analysis against Bhuvaji et al.
|-- src/
|   |-- __init__.py
|   |-- schemas/                   # Pydantic models for decisions, tasks, and alerts
|   |   |-- __init__.py
|   |   `-- models.py
|   |-- memory/                    # Temporal Knowledge Graph (NetworkX / Graph RAG)
|   |   `-- temporal_graph.py
|   |-- reasoning/                 # NLI Contradiction and Conflict Detection Engine
|   |   `-- nli_conflict_detector.py
|   |-- execution/                 # Human-in-the-loop review and tracker connectors
|   |   `-- tracker_service.py
|   |-- cron/                      # Reminder agent and weekly digest generator
|   |-- main.py                    # Pipeline entrypoint and end-to-end demonstration
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

To run the end-to-end multi-meeting conflict detection pipeline:
```bash
python -m src.main
```

---

## Future Roadmap with Respect to the Base Paper

1. **Summary-Level Hierarchical Clustering**: Implementing HDBSCAN/Agglomerative clustering on historical meeting transcripts as proposed in the base paper to categorize meetings by product area prior to graph insertion.
2. **Hybrid Dense-Sparse RAG**: Combining BM25 sparse keyword indices with dense sentence-transformers embeddings for low-latency lexical entity retrieval.
3. **Multi-Modal Voice Ingestion**: Integrating live local Whisper streaming directly from microphone input for real-time meeting transcription.
4. **Automated Supersession State Updates**: When a human supervisor approves a conflicting decision, the graph automatically marks the predecessor decision node as `superseded_by` with historical provenance.

---

## Author

**Satya Ram Dinkar**
* GitHub: [@SatyaRamDinkar](https://github.com/SatyaRamDinkar)
* Role: AI/ML Engineer
