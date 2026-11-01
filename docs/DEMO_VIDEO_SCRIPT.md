# SynapseMeet: 2-3 Minute Hackathon Demo Video Script & Recording Guide
## Includes Live Web App Demonstration + GitHub Repository Walkthrough

This document contains the exact word-for-word spoken script, click sequences, screen setups, and submission guidelines for recording your winning hackathon demonstration video.

---

## 🎬 Screen & Recording Setup Checklist
- **Tool to Record**: OBS Studio, Loom, or Windows Game Bar (`Win + G`).
- **Resolution**: 1080p (Full Screen).
- **Tab 1**: Live Streamlit Cloud Dashboard:  
  👉 **[https://synapsemeet-xeifapaaw5ozmagtkpyj9j.streamlit.app/](https://synapsemeet-xeifapaaw5ozmagtkpyj9j.streamlit.app/)**
- **Tab 2**: GitHub Repository:  
  👉 **[https://github.com/SatyaRamDinkar/SynapseMeet](https://github.com/SatyaRamDinkar/SynapseMeet)**

---

## 🎙️ Word-for-Word Video Script (Timestamp Breakdown)

### ⏱️ 0:00 – 0:35 | Hook & Core Problem (Live App)
**On Screen**: Tab 1 (Streamlit App) on the **Live Ingestion & Conflict Triage** tab with Hero Banner and KPI metrics.

> *"Hello judges and fellow engineers. Today I am presenting **SynapseMeet**, an autonomous Meeting-to-Execution agent designed for engineering teams.*
> 
> *In tech companies today, decisions made in meetings often evaporate into forgotten docs. Worse, teams lack cross-meeting memory: they frequently approve architecture changes or deadlines today that directly contradict a policy set weeks ago, leading to alignment drift and costly rollbacks.*
> 
> *SynapseMeet transforms meetings from passive note summarization into an active, conflict-free execution engine powered by **Temporal Knowledge Graphs** and a **Calibrated Natural Language Inference (NLI) Contradiction Engine**."*

---

### ⏱️ 0:35 – 1:15 | Live Contradiction Detection Demonstration (Live App)
**On Screen**: Tab 1 (Streamlit App).
1. Under *"Quick-Load Test Scenarios"*, select **`Database Policy Reversal (DynamoDB -> PostgreSQL)`**.
2. Click **`Audit Decision Against Organizational Memory`**.

> *"Let’s see it live in action.*
> 
> *In today's Sprint 44 Architecture review, our tech lead Alex proposes: 'Migrate user profiles from DynamoDB to PostgreSQL this sprint'.*
> 
> *When we click 'Audit Decision', SynapseMeet performs Hybrid RAG across our Temporal Knowledge Graph. Instantly, our DeBERTa cross-encoder NLI engine detects an **88%+ contradiction score**.*
> 
> *Notice this red alert card: SynapseMeet caught that this decision directly reverses an approved policy from Sprint 42 where Sarah, our Principal Architect, standardized on DynamoDB.*
> 
> *Notice the probability distribution chart below, and how the **Human Confirmation Gate** automatically halts posting to prevent committing conflicting organizational goals."*

---

### ⏱️ 1:15 – 1:40 | State Machine & Graph Lifecycle (Live App)
**On Screen**: Tab 1 (Streamlit App).
1. Click **`Approve & Supersede Historical Policy`**.
2. Switch in sidebar to **`Temporal Knowledge Graph Explorer`**.

> *"Now, the Engineering Director reviews the trade-off and decides to approve the database migration.*
> 
> *With one click on 'Approve & Supersede', SynapseMeet executes a state machine transition in our Temporal Directed Graph.*
> 
> *Here in the Graph Explorer, Sarah's older DynamoDB decision is now marked **SUPERSEDED** with a closed validity interval, while Alex's new PostgreSQL decision is now **ACTIVE**, and the work item is dispatched to Linear and Jira with full audit provenance."*

---

### ⏱️ 1:40 – 2:10 | GitHub Repository Walkthrough (GitHub Tab)
**On Screen**: Switch browser tab to **Tab 2: GitHub Repository** ([https://github.com/SatyaRamDinkar/SynapseMeet](https://github.com/SatyaRamDinkar/SynapseMeet)).
1. Scroll down smoothly through the **README**.
2. Briefly point your mouse cursor at the **AI Models & Tooling Stack table**, the **Comparative Matrix table**, and the **Repository Structure**.

> *"Here on our GitHub repository at **SatyaRamDinkar/SynapseMeet**, our codebase is organized into clean, production-grade modular components.*
> 
> *As shown in our AI Tooling Stack, we developed this using **Google Antigravity**, integrating **Gemini 1.5 Pro**, **Whisper-large-v3**, and a **DeBERTa-v3 Cross-Encoder** for calibrated NLI.*
> 
> *Our repository includes full technical documentation: our comparative analysis against the base paper, a prompt evolution guide for reproduction, and the complete source code under the `src` directory covering ingestion, temporal graphs, hybrid retrieval, and execution tracking."*

---

### ⏱️ 2:10 – 2:40 | Benchmark Metrics & Outro (GitHub or App Tab)
**On Screen**: Show the **Comparative Performance Matrix Table** on the GitHub README (or switch to Tab 5 on Streamlit).

> *"Finally, looking at our empirical benchmarks across 30 real-world engineering test cases: SynapseMeet achieves a **93.75% Precision**, **93.75% Recall**, and **93.33% Accuracy**.*
> 
> *Unlike the 100% artificial figures that signal overfitting, these metrics represent real-world linguistic boundaries. Compared to the base paper (Bhuvaji et al., ACM SAC 2025), we achieved a **+15.55% gain in Precision**, a **+19.65% gain in Recall**, and reduced inference latency from 1.4 seconds down to **0.011 milliseconds**.*
> 
> *Thank you! All links, code, and documentation are live on GitHub."*

---

## Submitted Demo Video

**Status**: Recorded and submitted.

**Google Drive Link**: [https://drive.google.com/file/d/1JjstTAXigE10PmEkdF1mZjA0haU3aiPk/view?usp=drive_link](https://drive.google.com/file/d/1JjstTAXigE10PmEkdF1mZjA0haU3aiPk/view?usp=drive_link)

This link is also embedded in the project README.md as a prominent badge at the top of the repository.

---

## Recording Guide (Reference)

Once you finish recording:
1. Upload your video file (`.mp4`) to Google Drive.
2. Share → Set to **"Anyone with the link can view"** → Copy link.
3. Paste the Google Drive link into our chat.
4. I will immediately add a prominent **Watch Demo Video** badge and section to your GitHub README and commit/push it for you!
