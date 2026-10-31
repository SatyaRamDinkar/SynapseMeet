import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.schemas.models import DecisionItem, ActionItem, MeetingTranscript, SpeakerUtterance
from src.clustering.meeting_clusterer import MeetingSummaryClusterer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.memory.temporal_graph import TemporalKnowledgeGraph
from src.reasoning.advanced_nli_detector import AdvancedNLIConflictDetector
from src.execution.tracker_service import TaskTrackerService
from src.evaluation.comprehensive_benchmark import ComprehensiveEvaluationHarness

# --- Page Configuration ---
st.set_page_config(
    page_title="SynapseMeet | Autonomous Meeting Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: bold; color: #0284C7; }
    .metric-label { font-size: 0.85rem; color: #64748B; }
    .alert-box { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 4px; margin: 15px 0; }
    .success-box { background-color: #F0FDF4; border-left: 5px solid #22C55E; padding: 15px; border-radius: 4px; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "graph" not in st.session_state:
    st.session_state.graph = TemporalKnowledgeGraph()
    # Add historical baseline
    d_hist = DecisionItem(
        decision_id="DEC_ARCH_01",
        topic="Database Storage",
        statement="We decided to standardize our data store on DynamoDB for all user profiles.",
        owner="Sarah (Principal Architect)",
        meeting_id="MTG_INFRA_01",
        timestamp=datetime.now() - timedelta(days=21),
        status="ACTIVE"
    )
    st.session_state.graph.add_decision(d_hist)

if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever()
    st.session_state.retriever.index_decisions([{
        "id": "DEC_ARCH_01",
        "topic": "Database Storage",
        "statement": "We decided to standardize our data store on DynamoDB for all user profiles.",
        "owner": "Sarah",
        "meeting_id": "MTG_INFRA_01",
        "timestamp": (datetime.now() - timedelta(days=21)).isoformat()
    }])

if "dispatched_tasks" not in st.session_state:
    st.session_state.dispatched_tasks = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚡ SynapseMeet")
    st.markdown("**Autonomous Meeting-to-Execution Agent**")
    st.caption("Cross-Meeting Temporal Memory & Calibrated NLI")
    st.divider()

    menu = st.radio("Navigation", [
        "Live Meeting Ingestion & Conflict Triage",
        "Temporal Knowledge Graph Explorer",
        "Meeting Clusters (ACM SAC 2025)",
        "Dispatched Work Items (Trackers)",
        "Quantitative Evaluation Benchmark"
    ])

    st.divider()
    st.markdown("### 📚 Base Paper")
    st.markdown("**A Retrieval-Augmented Framework For Meeting Insight Extraction**  \n*Bhuvaji et al., ACM SAC 2025*")
    st.divider()
    st.markdown("👨‍💻 **Author:** Satya Ram Dinkar  \n[@SatyaRamDinkar](https://github.com/SatyaRamDinkar)")

# =========================================================================
# TAB 1: LIVE MEETING INGESTION & CONFLICT TRIAGE
# =========================================================================
if menu == "Live Meeting Ingestion & Conflict Triage":
    st.markdown('<div class="main-title">Live Meeting Ingestion & Conflict Triage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ingest new meeting decisions, run Hybrid RAG, and verify contradictions against organizational memory.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("New Meeting Decision Intake")
        meeting_id = st.text_input("Meeting Identifier", value="MTG_INFRA_02 (Sprint 44 Arch Review)")
        topic = st.selectbox("Domain Topic", ["Database Storage", "Cloud Infrastructure", "Frontend & Mobile", "DevOps & Releases", "Security & Governance"])
        owner = st.text_input("Decision Owner / Tech Lead", value="Alex (Tech Lead)")
        
        sample_presets = {
            "Custom Input": "",
            "Database Reversal (DynamoDB -> PostgreSQL)": "Migrate user profiles from DynamoDB to PostgreSQL this sprint.",
            "Cloud Platform Shift (AWS -> GCP)": "Move container workloads from AWS to GCP Kubernetes engine.",
            "Frontend Disagreement (React Native -> Flutter)": "Rewrite the mobile codebase in Flutter.",
            "Release Policy Reversal (Deploy -> Freeze)": "Enforce code freeze starting this Friday for all teams."
        }
        selected_preset = st.selectbox("Quick Test Scenario Presets", list(sample_presets.keys()))
        
        default_stmt = sample_presets[selected_preset] if sample_presets[selected_preset] else "Migrate user profiles from DynamoDB to PostgreSQL this sprint."
        statement = st.text_area("Decision Statement", value=default_stmt, height=100)

        task_desc = st.text_input("Associated Action Item", value="Write schema migration and backfill scripts from DynamoDB to PostgreSQL")
        assignee = st.text_input("Task Assignee", value="Alex")

        evaluate_btn = st.button("Evaluate Decision against Cross-Meeting Memory", type="primary")

    with col2:
        st.subheader("Cross-Meeting Intelligence & NLI Audit")
        
        if evaluate_btn:
            new_decision = DecisionItem(
                decision_id=f"DEC_{int(datetime.now().timestamp())}",
                topic=topic,
                statement=statement,
                owner=owner,
                meeting_id=meeting_id,
                timestamp=datetime.now()
            )

            # Retrieve related active decisions from Temporal Graph
            active_decisions = st.session_state.graph.get_active_decisions()
            detector = AdvancedNLIConflictDetector()

            conflict_found = None
            for hist in active_decisions:
                hist_item = DecisionItem(
                    decision_id=hist["decision_id"],
                    topic=hist["topic"],
                    statement=hist["statement"],
                    owner=hist["owner"],
                    meeting_id=hist["meeting_id"],
                    timestamp=datetime.fromisoformat(hist["timestamp"])
                )
                conflict_found = detector.evaluate_pair(hist_item, new_decision)
                if conflict_found:
                    break

            if conflict_found:
                st.markdown(f"""
                <div class="alert-box">
                    <h4 style="color: #991B1B; margin-top:0;">🚨 CONFLICT DETECTED across meetings!</h4>
                    <p><b>Domain:</b> {topic}</p>
                    <p><b>Proposed Today ({meeting_id}):</b> "{statement}"</p>
                    <p><b>Contradicts Past Meeting ({conflict_found.historical_meeting_id}):</b> "{conflict_found.conflicting_decision.statement}"</p>
                    <p><b>Reasoning:</b> {conflict_found.explanation}</p>
                    <p><b>Contradiction Probability:</b> <span style="font-weight:bold; color:#DC2626;">{conflict_found.contradiction_score * 100:.2f}%</span></p>
                </div>
                """, unsafe_allow_html=True)

                st.warning("🛡️ Human Confirmation Gate: Work item posting halted to prevent committing conflicting goals.")
                
                col_ov1, col_ov2 = st.columns(2)
                with col_ov1:
                    if st.button("Approve & Supersede Historical Policy", type="primary"):
                        st.session_state.graph.add_decision(new_decision)
                        st.session_state.graph.supersede_decision(
                            old_decision_id=conflict_found.conflicting_decision.decision_id,
                            new_decision_id=new_decision.decision_id,
                            timestamp=datetime.now()
                        )
                        task_tracker = TaskTrackerService()
                        act = ActionItem(
                            task_id=f"TASK_{int(datetime.now().timestamp())}",
                            description=task_desc,
                            assignee=assignee,
                            deadline=datetime.now() + timedelta(days=7),
                            meeting_id=meeting_id
                        )
                        res = task_tracker.dispatch_action_item(act, confirmed_by_user=True)
                        st.session_state.dispatched_tasks.append(res)
                        st.success(f"State Transitioned: Historical decision marked SUPERSEDED. Task {res['external_issue_id']} dispatched!")
                        st.rerun()
                with col_ov2:
                    if st.button("Reject New Proposal"):
                        st.info("Proposal rejected. Historical policy maintained intact.")
            else:
                st.markdown("""
                <div class="success-box">
                    <h4 style="color: #166534; margin-top:0;">✅ Zero Contradictions Detected</h4>
                    <p>Decision conforms with all historical organizational commitments and active policies.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Confirm and Dispatch Task to Tracker"):
                    st.session_state.graph.add_decision(new_decision)
                    task_tracker = TaskTrackerService()
                    act = ActionItem(
                        task_id=f"TASK_{int(datetime.now().timestamp())}",
                        description=task_desc,
                        assignee=assignee,
                        deadline=datetime.now() + timedelta(days=7),
                        meeting_id=meeting_id
                    )
                    res = task_tracker.dispatch_action_item(act, confirmed_by_user=True)
                    st.session_state.dispatched_tasks.append(res)
                    st.success(f"Task {res['external_issue_id']} posted successfully!")
                    st.rerun()

# =========================================================================
# TAB 2: TEMPORAL KNOWLEDGE GRAPH EXPLORER
# =========================================================================
elif menu == "Temporal Knowledge Graph Explorer":
    st.markdown('<div class="main-title">Temporal Knowledge Graph Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Audit cross-meeting organizational memory, state transitions, and validity lifecycles.</div>', unsafe_allow_html=True)

    decisions_data = []
    for node_id, attrs in st.session_state.graph.graph.nodes(data=True):
        if attrs.get("type") == "Decision":
            decisions_data.append({
                "Decision ID": attrs.get("decision_id"),
                "Topic": attrs.get("topic", "").title(),
                "Statement": attrs.get("statement"),
                "Owner": attrs.get("owner"),
                "Meeting": attrs.get("meeting_id"),
                "Status": attrs.get("status"),
                "Valid From": attrs.get("valid_from", "")[:19],
                "Superseded By": attrs.get("superseded_by") or "None (Current)"
            })

    if decisions_data:
        df = pd.DataFrame(decisions_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No decisions recorded yet.")

# =========================================================================
# TAB 3: MEETING CLUSTERS (ACM SAC 2025)
# =========================================================================
elif menu == "Meeting Clusters (ACM SAC 2025)":
    st.markdown('<div class="main-title">Meeting Clusters (ACM SAC 2025)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hierarchical Agglomerative TF-IDF Clustering grouping historical transcripts into semantic domains.</div>', unsafe_allow_html=True)

    clusterer = MeetingSummaryClusterer(n_clusters=3)
    sample_meetings = [
        MeetingTranscript(
            meeting_id="MTG_INFRA_01",
            title="Infrastructure & Cloud Storage Baseline",
            date=datetime.now() - timedelta(days=21),
            participants=["Sarah (Architect)", "Dave (DevOps)"],
            utterances=[SpeakerUtterance(speaker_id="Sarah", start_time=0.0, end_time=10.0, text="Standardize on DynamoDB.")],
            summary="Standardized data storage layer on DynamoDB and planned AWS cloud deployment."
        ),
        MeetingTranscript(
            meeting_id="MTG_FRONTEND_01",
            title="Client Applications and UI Framework Review",
            date=datetime.now() - timedelta(days=14),
            participants=["Frank (Mobile Lead)", "Grace (Frontend)"],
            utterances=[SpeakerUtterance(speaker_id="Frank", start_time=0.0, end_time=10.0, text="Selected React Native.")],
            summary="Finalized React Native for cross-platform mobile client implementation."
        ),
        MeetingTranscript(
            meeting_id="MTG_INFRA_02",
            title="Sprint 44 Architecture & Scalability Review",
            date=datetime.now(),
            participants=["Alex (Tech Lead)", "Sarah (Architect)"],
            utterances=[SpeakerUtterance(speaker_id="Alex", start_time=0.0, end_time=10.0, text="Migrate to PostgreSQL.")],
            summary="Proposed database migration to PostgreSQL to support complex relational queries."
        )
    ]
    clustered = clusterer.fit_clusters(sample_meetings)

    c_data = [{
        "Meeting ID": m.meeting_id,
        "Meeting Title": m.title,
        "Cluster ID": m.cluster_id,
        "Semantic Domain Topic": m.cluster_topic,
        "Participants": ", ".join(m.participants)
    } for m in clustered]

    st.table(pd.DataFrame(c_data))

# =========================================================================
# TAB 4: DISPATCHED WORK ITEMS
# =========================================================================
elif menu == "Dispatched Work Items (Trackers)":
    st.markdown('<div class="main-title">Dispatched Work Items (Trackers)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Synchronized tasks committed to Linear / Jira / GitHub after passing human verification.</div>', unsafe_allow_html=True)

    if st.session_state.dispatched_tasks:
        st.dataframe(pd.DataFrame(st.session_state.dispatched_tasks), use_container_width=True)
    else:
        st.info("No tasks dispatched yet. Run a decision through the Live Ingestion tab and confirm.")

# =========================================================================
# TAB 5: QUANTITATIVE BENCHMARK
# =========================================================================
elif menu == "Quantitative Evaluation Benchmark":
    st.markdown('<div class="main-title">Quantitative Evaluation Benchmark (v2.0)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical verification across 25 multi-domain engineering meeting pairs.</div>', unsafe_allow_html=True)

    if st.button("Run Live Benchmark Suite", type="primary"):
        with st.spinner("Executing 25-pair evaluation across 8 engineering domains..."):
            harness = ComprehensiveEvaluationHarness()
            res = harness.run_benchmark()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Precision", f"{res['precision']*100:.2f}%", delta="Target: >= 90%")
        with c2:
            st.metric("Recall (Sensitivity)", f"{res['recall']*100:.2f}%", delta="Target: >= 90%")
        with c3:
            st.metric("F1-Score", f"{res['f1_score']*100:.2f}%", delta="Target: >= 90%")
        with c4:
            st.metric("Mean Latency", f"{res['mean_latency_ms']} ms", delta="Target: < 50ms")

        st.subheader("Confusion Matrix & Statistical Breakdown")
        breakdown_df = pd.DataFrame([
            {"Metric": "Total Evaluation Pairs", "Result": res["total_samples"], "Target": ">= 20"},
            {"Metric": "True Positives (Conflicts Caught)", "Result": res["true_positives"], "Target": "14"},
            {"Metric": "False Positives (Erroneous Alerts)", "Result": res["false_positives"], "Target": "0"},
            {"Metric": "True Negatives (Valid Agreements)", "Result": res["true_negatives"], "Target": "11"},
            {"Metric": "False Negatives (Missed Conflicts)", "Result": res["false_negatives"], "Target": "0"},
            {"Metric": "Specificity", "Result": f"{res['specificity']*100:.2f}%", "Target": ">= 90%"},
            {"Metric": "Overall Accuracy", "Result": f"{res['accuracy']*100:.2f}%", "Target": ">= 95%"},
            {"Metric": "P95 Latency", "Result": f"{res['p95_latency_ms']} ms", "Target": "< 100 ms"}
        ])
        st.table(breakdown_df)
