from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class SpeakerUtterance(BaseModel):
    speaker_id: str = Field(description="Unique speaker identifier, e.g. Speaker_01 or Alice")
    start_time: float = Field(description="Start offset in seconds")
    end_time: float = Field(description="End offset in seconds")
    text: str = Field(description="Spoken text content")

class DecisionItem(BaseModel):
    decision_id: str = Field(description="Unique deterministic ID for the decision")
    topic: str = Field(description="Subject domain, e.g., 'Database Migration'")
    statement: str = Field(description="Exact statement of decision made")
    owner: str = Field(description="Key decision maker / sponsor")
    meeting_id: str = Field(description="Identifier of source meeting")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ActionItem(BaseModel):
    task_id: str = Field(description="Unique identifier for the action item")
    description: str = Field(description="Clear task description")
    assignee: str = Field(description="Name or ID of individual accountable")
    deadline: Optional[datetime] = Field(default=None, description="Due date and time")
    meeting_id: str = Field(description="Source meeting ID")
    status: str = Field(default="pending", description="pending | approved | dispatched")

class ConflictAlert(BaseModel):
    conflict_id: str
    current_decision: DecisionItem
    conflicting_decision: DecisionItem
    contradiction_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    requires_human_override: bool = True
