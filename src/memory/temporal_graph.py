import networkx as nx
from typing import List, Optional, Dict
from src.schemas.models import DecisionItem, ActionItem

class TemporalKnowledgeGraph:
    """
    In-memory Temporal Knowledge Graph representing organizational commitments,
    decisions, and dependencies across multi-meeting timelines.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_decision(self, decision: DecisionItem):
        node_id = f"decision:{decision.decision_id}"
        self.graph.add_node(
            node_id,
            type="Decision",
            topic=decision.topic.lower(),
            statement=decision.statement,
            owner=decision.owner,
            meeting_id=decision.meeting_id,
            timestamp=decision.timestamp.isoformat()
        )
        # Link owner to decision
        owner_id = f"person:{decision.owner.lower()}"
        self.graph.add_node(owner_id, type="Person", name=decision.owner)
        self.graph.add_edge(owner_id, node_id, relation="SPONSORED")
        return node_id

    def add_action_item(self, action: ActionItem):
        node_id = f"task:{action.task_id}"
        self.graph.add_node(
            node_id,
            type="Task",
            description=action.description,
            status=action.status,
            deadline=action.deadline.isoformat() if action.deadline else None,
            meeting_id=action.meeting_id
        )
        assignee_id = f"person:{action.assignee.lower()}"
        self.graph.add_node(assignee_id, type="Person", name=action.assignee)
        self.graph.add_edge(assignee_id, node_id, relation="ASSIGNED_TO")
        return node_id

    def get_decisions_by_topic(self, topic: str) -> List[Dict]:
        topic = topic.lower()
        results = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "Decision":
                if topic in attrs.get("topic", "") or attrs.get("topic", "") in topic:
                    results.append({"id": node_id, **attrs})
        return results
