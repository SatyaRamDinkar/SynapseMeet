from typing import List
from src.schemas.models import ActionItem

class TaskTrackerService:
    """
    Guarded tracker integration (Linear, GitHub Issues, Jira).
    Requires explicit human confirmation before dispatching issues.
    """
    def __init__(self):
        self.dispatched_tasks = []

    def dispatch_action_item(self, action: ActionItem, confirmed_by_user: bool = False) -> dict:
        if not confirmed_by_user:
            return {
                "task_id": action.task_id,
                "status": "blocked",
                "reason": "Human confirmation required before posting to tracker."
            }

        action.status = "dispatched"
        record = {
            "task_id": action.task_id,
            "external_issue_id": f"LINEAR-{action.task_id.upper()}",
            "description": action.description,
            "assignee": action.assignee,
            "deadline": action.deadline.isoformat() if action.deadline else None,
            "status": "synced"
        }
        self.dispatched_tasks.append(record)
        return record
