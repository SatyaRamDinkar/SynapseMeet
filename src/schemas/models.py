from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class SpeakerUtterance(BaseModel):
    speaker_id: str = Field(description="Unique speaker identifier, e.g. Speaker_01 or Alice")
    start_time: float = Field(description="Start offset in seconds")
    end_time: float = Field(description="End offset in seconds")
    text: str = Field(description="Spoken text content")

class MeetingTranscript(BaseModel):
    meeting_id: str
    title: str
    date: datetime
    participants: List[str]
    utterances: List[SpeakerUtterance]
    summary: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_topic: Optional[str] = None

class DecisionItem(BaseModel):
    decision_id: str = Field(description="Unique deterministic ID for the decision")
    topic: str = Field(description="Subject domain, e.g., 'Database Architecture'")
    statement: str = Field(description="Exact statement of decision made")
    owner: str = Field(description="Key decision maker or sponsor")
    meeting_id: str = Field(description="Identifier of source meeting")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = Field(default="ACTIVE", description="ACTIVE | SUPERSEDED | OVERRIDDEN")
    superseded_by: Optional[str] = Field(default=None, description="ID of overriding decision")

class ActionItem(BaseModel):
    task_id: str = Field(description="Unique identifier for the action item")
    description: str = Field(description="Clear task description")
    assignee: str = Field(description="Name or ID of individual accountable")
    deadline: Optional[datetime] = Field(default=None, description="Due date and time")
    meeting_id: str = Field(description="Source meeting ID")
    status: str = Field(default="pending", description="pending | approved | dispatched")
    external_issue_id: Optional[str] = None

class ConflictAlert(BaseModel):
    conflict_id: str
    current_decision: DecisionItem
    conflicting_decision: DecisionItem
    contradiction_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral_score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str
    historical_meeting_id: str
    requires_human_override: bool = True
