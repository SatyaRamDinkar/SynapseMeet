import sys
import io
from datetime import datetime, timedelta

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.schemas.models import DecisionItem, ActionItem, MeetingTranscript, SpeakerUtterance
from src.clustering.meeting_clusterer import MeetingSummaryClusterer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.memory.temporal_graph import TemporalKnowledgeGraph
from src.reasoning.advanced_nli_detector import AdvancedNLIConflictDetector
from src.execution.tracker_service import TaskTrackerService
from src.evaluation.comprehensive_benchmark import ComprehensiveEvaluationHarness

console = Console()

def run_pipeline():
    console.print(Panel.fit(
        "[bold cyan]SynapseMeet: Autonomous Meeting-to-Execution Intelligence Engine[/bold cyan]\n"
        "[dim]Extended from ACM SAC 2025 Base Paper (Bhuvaji et al.) with Temporal Knowledge Graphs & Calibrated NLI[/dim]"
    ))

    # =========================================================================
    # STAGE 1: MEETING INGESTION & SUMMARY CLUSTERING (Bhuvaji et al. SAC 2025)
    # =========================================================================
    console.print("\n[bold yellow]Stage 1: Meeting Ingestion & Agglomerative Summary Clustering[/bold yellow]")
    clusterer = MeetingSummaryClusterer(n_clusters=3)

    m1 = MeetingTranscript(
        meeting_id="MTG_INFRA_01",
        title="Infrastructure & Cloud Storage Baseline",
        date=datetime.now() - timedelta(days=21),
        participants=["Sarah (Architect)", "Dave (DevOps)"],
        utterances=[
            SpeakerUtterance(speaker_id="Sarah", start_time=0.0, end_time=12.0, text="We decided to standardize our data store on DynamoDB for all user profiles."),
            SpeakerUtterance(speaker_id="Dave", start_time=13.0, end_time=25.0, text="Agreed, we will provision DynamoDB tables in AWS us-east-1.")
        ],
        summary="Standardized data storage layer on DynamoDB and planned AWS cloud deployment."
    )

    m2 = MeetingTranscript(
        meeting_id="MTG_FRONTEND_01",
        title="Client Applications and UI Framework Review",
        date=datetime.now() - timedelta(days=14),
        participants=["Frank (Mobile Lead)", "Grace (Frontend)"],
        utterances=[
            SpeakerUtterance(speaker_id="Frank", start_time=0.0, end_time=10.0, text="Our team selected React Native for the multiplatform client application."),
        ],
        summary="Finalized React Native for cross-platform mobile client implementation."
    )

    m3 = MeetingTranscript(
        meeting_id="MTG_INFRA_02",
        title="Sprint 44 Architecture & Scalability Review",
        date=datetime.now(),
        participants=["Alex (Tech Lead)", "Sarah (Architect)"],
        utterances=[
            SpeakerUtterance(speaker_id="Alex", start_time=0.0, end_time=15.0, text="We should migrate user profiles from DynamoDB to PostgreSQL this sprint."),
        ],
        summary="Proposed database migration to PostgreSQL to support complex relational queries."
    )

    clustered_meetings = clusterer.fit_clusters([m1, m2, m3])

    cluster_table = Table(title="Historical Meeting Clusters (ACM SAC 2025 Methodology)")
    cluster_table.add_column("Meeting ID", style="cyan")
    cluster_table.add_column("Meeting Title", style="white")
    cluster_table.add_column("Cluster ID", style="magenta")
    cluster_table.add_column("Assigned Domain Topic", style="green")

    for m in clustered_meetings:
        cluster_table.add_row(m.meeting_id, m.title, str(m.cluster_id), m.cluster_topic or "General")
    console.print(cluster_table)

    # =========================================================================
    # STAGE 2: TEMPORAL KNOWLEDGE GRAPH & HYBRID RETRIEVAL INDEXING
    # =========================================================================
    console.print("\n[bold yellow]Stage 2: Indexing Decisions into Temporal Graph & Hybrid RAG[/bold yellow]")
    graph = TemporalKnowledgeGraph()
    retriever = HybridRetriever()

    # Ingest historical decision into graph
    d_historical = DecisionItem(
        decision_id="DEC_ARCH_01",
        topic="Database Storage",
        statement="We decided to standardize our data store on DynamoDB for all user profiles.",
        owner="Sarah",
        meeting_id="MTG_INFRA_01",
        timestamp=datetime.now() - timedelta(days=21),
        status="ACTIVE"
    )
    graph.add_decision(d_historical)

    # Index into hybrid retriever (BM25 + Semantic)
    retriever.index_decisions([
        {
            "id": d_historical.decision_id,
            "topic": d_historical.topic,
            "statement": d_historical.statement,
            "owner": d_historical.owner,
            "meeting_id": d_historical.meeting_id,
            "timestamp": d_historical.timestamp.isoformat()
        }
    ])
    console.print(f"[green][OK][/green] Ingested Decision '{d_historical.statement}' (Owner: {d_historical.owner}) into Temporal Graph")

    # =========================================================================
    # STAGE 3: NEW DECISION PARSING & HYBRID RAG RETRIEVAL
    # =========================================================================
    console.print("\n[bold yellow]Stage 3: New Meeting Transcript Ingestion & Decision Extraction[/bold yellow]")
    d_new = DecisionItem(
        decision_id="DEC_ARCH_02",
        topic="Database Storage",
        statement="Migrate user profiles from DynamoDB to PostgreSQL this sprint.",
        owner="Alex",
        meeting_id="MTG_INFRA_02",
        timestamp=datetime.now()
    )
    console.print(f"[cyan]Parsed New Proposed Decision:[/cyan] '{d_new.statement}' (Meeting: {d_new.meeting_id})")

    # Hybrid Search across historical memory
    search_results = retriever.search(query=f"{d_new.topic} {d_new.statement}", top_k=2)
    console.print(f"[dim]Hybrid RAG (BM25 + Dense RRF) retrieved {len(search_results)} relevant historical decisions.[/dim]")

    # =========================================================================
    # STAGE 4: ADVANCED NLI CONTRADICTION & CROSS-MEETING CONFLICT DETECTION
    # =========================================================================
    console.print("\n[bold yellow]Stage 4: Calibrated Natural Language Inference (NLI) Contradiction Scoring[/bold yellow]")
    nli_detector = AdvancedNLIConflictDetector()
    conflict = nli_detector.evaluate_pair(d_historical, d_new)

    if conflict:
        console.print(Panel(
            f"[bold red][ALERT] CROSS-MEETING CONTRADICTION DETECTED![/bold red]\n\n"
            f"[bold]Domain Topic:[/bold] {d_new.topic}\n"
            f"[bold]New Proposed Stance ({d_new.meeting_id}):[/bold] {d_new.statement}\n"
            f"[bold]Historical Active Policy ({d_historical.meeting_id}):[/bold] {d_historical.statement}\n\n"
            f"[bold]NLI Entailment Probability:[/bold] {conflict.entailment_score * 100:.2f}%\n"
            f"[bold]NLI Neutral Probability:[/bold]    {conflict.neutral_score * 100:.2f}%\n"
            f"[bold]NLI Contradiction Score:[/bold]   {conflict.contradiction_score * 100:.2f}%\n\n"
            f"[bold]Diagnostic Root Cause:[/bold] {conflict.explanation}",
            border_style="red"
        ))

    # =========================================================================
    # STAGE 5: HUMAN CONFIRMATION GATE & TEMPORAL STATE TRANSITION
    # =========================================================================
    console.print("[bold yellow]Stage 5: Human Confirmation Gate & Temporal Graph Transition[/bold yellow]")
    console.print("[dim]Simulating Team Lead manual review: Operator explicitly approves database transition.[/dim]")

    # Transition historical node state in Temporal Graph
    graph.add_decision(d_new)
    graph.supersede_decision(old_decision_id=d_historical.decision_id, new_decision_id=d_new.decision_id, timestamp=datetime.now())
    console.print(f"[green][STATE UPDATED][/green] Decision '{d_historical.decision_id}' transitioned to state: [bold red]SUPERSEDED[/bold red]")
    console.print(f"[green][STATE UPDATED][/green] Decision '{d_new.decision_id}' marked as: [bold green]ACTIVE[/bold green]")

    # Dispatch to Tracker
    tracker = TaskTrackerService()
    task = ActionItem(
        task_id="TASK_809",
        description="Write schema migration and backfill scripts from DynamoDB to PostgreSQL",
        assignee="Alex",
        deadline=datetime.now() + timedelta(days=7),
        meeting_id="MTG_INFRA_02"
    )
    dispatch_res = tracker.dispatch_action_item(task, confirmed_by_user=True)

    task_table = Table(title="Dispatched Tracker Work Items")
    task_table.add_column("Task ID", style="cyan")
    task_table.add_column("External Key", style="magenta")
    task_table.add_column("Description", style="white")
    task_table.add_column("Assignee", style="green")
    task_table.add_column("Status", style="yellow")
    task_table.add_row(dispatch_res["task_id"], dispatch_res["external_issue_id"], dispatch_res["description"], dispatch_res["assignee"], dispatch_res["status"])
    console.print(task_table)

    # =========================================================================
    # STAGE 6: COMPREHENSIVE BENCHMARK EVALUATION (25 MULTI-DOMAIN SAMPLES)
    # =========================================================================
    console.print("\n[bold yellow]Stage 6: Running Comprehensive 25-Sample Multi-Domain Evaluation Benchmark[/bold yellow]")
    benchmark = ComprehensiveEvaluationHarness()
    metrics = benchmark.run_benchmark()

    bench_table = Table(title="SynapseMeet Comprehensive Evaluation Benchmark (v2.0)")
    bench_table.add_column("Metric Dimension", style="bold cyan")
    bench_table.add_column("Score / Result", style="bold green")
    bench_table.add_column("Standard Target", style="dim white")

    bench_table.add_row("Total Evaluation Samples", str(metrics["total_samples"]), ">= 20")
    bench_table.add_row("True Positives (Conflicts Caught)", str(metrics["true_positives"]), "14")
    bench_table.add_row("False Positives (Erroneous Alerts)", str(metrics["false_positives"]), "0")
    bench_table.add_row("True Negatives (Valid Agreements)", str(metrics["true_negatives"]), "11")
    bench_table.add_row("False Negatives (Missed Conflicts)", str(metrics["false_negatives"]), "0")
    bench_table.add_row("Precision", f"{metrics['precision'] * 100:.2f}%", ">= 90%")
    bench_table.add_row("Recall (Sensitivity)", f"{metrics['recall'] * 100:.2f}%", ">= 90%")
    bench_table.add_row("Specificity", f"{metrics['specificity'] * 100:.2f}%", ">= 90%")
    bench_table.add_row("F1-Score", f"{metrics['f1_score'] * 100:.2f}%", ">= 90%")
    bench_table.add_row("Overall Accuracy", f"{metrics['accuracy'] * 100:.2f}%", ">= 95%")
    bench_table.add_row("Mean Inference Latency", f"{metrics['mean_latency_ms']} ms", "< 50 ms")
    bench_table.add_row("P95 Latency", f"{metrics['p95_latency_ms']} ms", "< 100 ms")
    console.print(bench_table)

    console.print("\n[bold green][COMPLETE] SynapseMeet upgraded pipeline and 25-pair multi-domain evaluation executed successfully![/bold green]")

if __name__ == "__main__":
    run_pipeline()
