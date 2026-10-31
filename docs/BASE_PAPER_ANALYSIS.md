# Academic Foundation & Base Paper Reference

## Primary Base Paper
**Title**: *A Retrieval-Augmented Framework For Meeting Insight Extraction*  
**Authors**: Sartaj Bhuvaji, Prachitee Chouhan, Madhuroopa Irukulla, Jay Singhvi, Wan D. Bae, Shayma Alkobaisi  
**Venue**: ACM Symposium on Applied Computing (**SAC 2025**)  
**Affiliations**: Seattle University & Collaborating Research Institutions  

---

## 🔬 Theoretical Connection & Extension in SynapseMeet

The base paper addresses information overload in collaborative organizations by designing an intelligent middleware layer using **Retrieval-Augmented Generation (RAG)** combined with summary clustering to recollect vital decisions, constraints, and action items from historical meeting transcripts.

### How SynapseMeet Extends the Base Paper:

| Dimension | Base Paper (*Bhuvaji et al., SAC 2025*) | **SynapseMeet (Our Hackathon Contribution)** |
| :--- | :--- | :--- |
| **Retrieval Paradigm** | Standard semantic chunk retrieval & summary clustering | **Temporal Knowledge Graph + Graph RAG** with valid-time interval edges |
| **Insight Depth** | Retrospective Q&A & historical insight extraction | **Proactive Contradiction Detection** using cross-meeting Natural Language Inference (NLI) |
| **Execution** | Passive information retrieval interface | **Autonomous Guarded Execution**: Dispatches confirmed issues to Linear / Jira / GitHub |
| **Human-in-the-Loop** | Manual reading of retrieved excerpts | **Interactive Conflict Review Card** before state commits |

---

## 📑 Formal Citation (BibTeX)

```bibtex
@inproceedings{bhuvaji2025retrieval,
  title={A Retrieval-Augmented Framework For Meeting Insight Extraction},
  author={Bhuvaji, Sartaj and Chouhan, Prachitee and Irukulla, Madhuroopa and Singhvi, Jay and Bae, Wan D. and Alkobaisi, Shayma},
  booktitle={Proceedings of the ACM Symposium on Applied Computing (SAC '25)},
  year={2025},
  publisher={ACM}
}
```
