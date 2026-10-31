from typing import Optional, Tuple
from src.schemas.models import DecisionItem, ConflictAlert

class NLIConflictDetector:
    """
    Computes semantic contradiction probabilities between newly proposed decisions
    and historical decisions stored in the temporal knowledge graph.
    """
    def __init__(self, contradiction_threshold: float = 0.65):
        self.threshold = contradiction_threshold

    def evaluate_pair(self, past_decision: DecisionItem, new_decision: DecisionItem) -> Optional[ConflictAlert]:
        """
        Calculates contradiction likelihood. In production, this utilizes
        HuggingFace cross-encoder/nli-deberta-v3-base or an LLM judge.
        """
        # Heuristic / semantic rule evaluation for rapid demonstration
        past_stmt = past_decision.statement.lower()
        new_stmt = new_decision.statement.lower()

        # Keyword opposition checks for database / architecture / cloud / framework choices
        opposing_pairs = [
            ("postgresql", "dynamodb"),
            ("aws", "gcp"),
            ("azure", "aws"),
            ("react", "vue"),
            ("remote", "in-office"),
            ("freeze", "deploy"),
            ("delay", "accelerate")
        ]

        contradiction_detected = False
        detected_terms = ()

        for term_a, term_b in opposing_pairs:
            if (term_a in past_stmt and term_b in new_stmt) or (term_b in past_stmt and term_a in new_stmt):
                contradiction_detected = True
                detected_terms = (term_a, term_b)
                break

        if contradiction_detected or ("postpone" in new_stmt and "immediate" in past_stmt):
            contradiction_score = 0.88
            explanation = (
                f"Historical decision in {past_decision.meeting_id} set policy: '{past_decision.statement}'. "
                f"New decision in {new_decision.meeting_id} changes stance to: '{new_decision.statement}'."
            )
            return ConflictAlert(
                conflict_id=f"conflict_{past_decision.decision_id}_{new_decision.decision_id}",
                current_decision=new_decision,
                conflicting_decision=past_decision,
                contradiction_score=contradiction_score,
                explanation=explanation,
                requires_human_override=True
            )
        return None
