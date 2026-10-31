import networkx as nx
from typing import List, Optional, Dict
from datetime import datetime
from src.schemas.models import DecisionItem, ActionItem

class TemporalKnowledgeGraph:
    """
    Temporal Directed Knowledge Graph representing organizational commitments,
    decisions, and state machine lifecycles with valid-time interval tracking.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_decision(self, decision: DecisionItem) -> str:
        node_id = f"decision:{decision.decision_id}"
        self.graph.add_node(
            node_id,
            type="Decision",
            decision_id=decision.decision_id,
            topic=decision.topic.lower(),
            statement=decision.statement,
            owner=decision.owner,
            meeting_id=decision.meeting_id,
            timestamp=decision.timestamp.isoformat(),
            status=decision.status,
            valid_from=decision.timestamp.isoformat(),
            valid_to=None,
            superseded_by=decision.superseded_by
        )
        owner_id = f"person:{decision.owner.lower()}"
        self.graph.add_node(owner_id, type="Person", name=decision.owner)
        self.graph.add_edge(owner_id, node_id, relation="SPONSORED", timestamp=decision.timestamp.isoformat())
        return node_id

    def supersede_decision(self, old_decision_id: str, new_decision_id: str, timestamp: datetime):
        """
        State Machine Transition: Marks historical decision as SUPERSEDED,
        closing its valid_to interval and creating a SUPERSEDES directed edge.
        """
        old_node = f"decision:{old_decision_id}"
        new_node = f"decision:{new_decision_id}"

        if old_node in self.graph.nodes:
            self.graph.nodes[old_node]["status"] = "SUPERSEDED"
            self.graph.nodes[old_node]["valid_to"] = timestamp.isoformat()
            self.graph.nodes[old_node]["superseded_by"] = new_decision_id

        if old_node in self.graph.nodes and new_node in self.graph.nodes:
            self.graph.add_edge(new_node, old_node, relation="SUPERSEDES", timestamp=timestamp.isoformat())

    def get_active_decisions(self) -> List[Dict]:
        """Returns only currently active, non-superseded decisions."""
        results = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "Decision" and attrs.get("status") == "ACTIVE":
                results.append({"id": node_id, **attrs})
        return results

    def get_decisions_by_topic(self, topic: str, include_superseded: bool = False) -> List[Dict]:
        topic = topic.lower()
        results = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "Decision":
                if not include_superseded and attrs.get("status") != "ACTIVE":
                    continue
                if topic in attrs.get("topic", "") or attrs.get("topic", "") in topic:
                    results.append({"id": node_id, **attrs})
        return results

    def add_action_item(self, action: ActionItem) -> str:
        node_id = f"task:{action.task_id}"
        self.graph.add_node(
            node_id,
            type="Task",
            task_id=action.task_id,
            description=action.description,
            status=action.status,
            deadline=action.deadline.isoformat() if action.deadline else None,
            meeting_id=action.meeting_id
        )
        assignee_id = f"person:{action.assignee.lower()}"
        self.graph.add_node(assignee_id, type="Person", name=action.assignee)
        self.graph.add_edge(assignee_id, node_id, relation="ASSIGNED_TO")
        return node_id
