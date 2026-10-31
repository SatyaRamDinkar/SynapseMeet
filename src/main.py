import sys
import io

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.schemas.models import DecisionItem, ActionItem
from src.memory.temporal_graph import TemporalKnowledgeGraph
from src.reasoning.nli_conflict_detector import NLIConflictDetector
from src.execution.tracker_service import TaskTrackerService

console = Console()

def run_demo():
    console.print(Panel.fit("[bold cyan]SynapseMeet: Meeting-to-Execution Agent Pipeline[/bold cyan]\n[dim]Autonomous Cross-Meeting Memory & NLI Conflict Detection Engine[/dim]"))

    # 1. Initialize Memory, Reasoning, and Execution
    graph = TemporalKnowledgeGraph()
    detector = NLIConflictDetector()
    tracker = TaskTrackerService()

    # 2. Historical Meeting Ingestion (Meeting 1 - Two weeks ago)
    past_decision = DecisionItem(
        decision_id="DEC_001",
        topic="Database Architecture",
        statement="We will use DynamoDB for all user profile storage until Q3 next year.",
        owner="Sarah (Principal Architect)",
        meeting_id="Sprint-42-Retro",
        timestamp=datetime.utcnow() - timedelta(days=14)
    )
    graph.add_decision(past_decision)
    console.print(f"[green][OK] Ingested historical decision from {past_decision.meeting_id}:[/green] '{past_decision.statement}'")

    # 3. New Meeting Ingestion (Meeting 2 - Today)
    console.print("\n[yellow]>> Ingesting new meeting transcript (Today's Architecture Review)...[/yellow]")
    new_decision = DecisionItem(
        decision_id="DEC_089",
        topic="Database Architecture",
        statement="Migrate user profiles to PostgreSQL by the end of this sprint.",
        owner="Alex (Tech Lead)",
        meeting_id="Arch-Review-Nov",
        timestamp=datetime.utcnow()
    )

    new_action = ActionItem(
        task_id="TASK_501",
        description="Write migration script from DynamoDB to PostgreSQL",
        assignee="Alex",
        deadline=datetime.utcnow() + timedelta(days=5),
        meeting_id="Arch-Review-Nov"
    )

    # 4. Conflict & Contradiction Evaluation
    console.print("\n[bold magenta]Querying Temporal Knowledge Graph & NLI Conflict Engine...[/bold magenta]")
    historical_decisions = graph.get_decisions_by_topic(new_decision.topic)

    conflict_alert = None
    for hist in historical_decisions:
        hist_decision_obj = DecisionItem(
            decision_id=hist["id"].replace("decision:", ""),
            topic=hist["topic"],
            statement=hist["statement"],
            owner=hist["owner"],
            meeting_id=hist["meeting_id"],
            timestamp=datetime.fromisoformat(hist["timestamp"])
        )
        conflict_alert = detector.evaluate_pair(hist_decision_obj, new_decision)
        if conflict_alert:
            break

    # 5. Display Conflict Alert if detected
    if conflict_alert:
        console.print(Panel(
            f"[bold red][ALERT] CONFLICT DETECTED across meetings![/bold red]\n\n"
            f"[bold]Topic:[/bold] {new_decision.topic}\n"
            f"[bold]New Proposed Decision ({new_decision.meeting_id}):[/bold] {new_decision.statement}\n"
            f"[bold]Contradicts Prior Decision ({past_decision.meeting_id}):[/bold] {past_decision.statement}\n"
            f"[bold]Contradiction Confidence:[/bold] {conflict_alert.contradiction_score * 100:.1f}%\n"
            f"[bold]Explanation:[/bold] {conflict_alert.explanation}",
            border_style="red"
        ))

    # 6. Human Confirmation Gate
    console.print("\n[bold blue]Human-in-the-Loop Confirmation Gate[/bold blue]")
    console.print("System halts automatic posting to avoid committing conflicting goals.")
    console.print("[dim]Simulating user review & approval with override flag = True...[/dim]")

    # User confirms the override
    result = tracker.dispatch_action_item(new_action, confirmed_by_user=True)
    
    table = Table(title="Dispatched Tracker Issues (Linear / Jira)")
    table.add_column("Task ID", style="cyan")
    table.add_column("External Issue", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Assignee", style="green")
    table.add_column("Status", style="bold yellow")
    table.add_row(result["task_id"], result["external_issue_id"], result["description"], result["assignee"], result["status"])
    console.print(table)

    console.print("\n[bold green][SUCCESS] Pipeline demonstration completed successfully![/bold green]")

if __name__ == "__main__":
    run_demo()
