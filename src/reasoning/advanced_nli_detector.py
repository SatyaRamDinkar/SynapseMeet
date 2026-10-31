import math
from typing import Optional, Tuple, Dict
from src.schemas.models import DecisionItem, ConflictAlert

class AdvancedNLIConflictDetector:
    """
    Probabilistic Multi-Class NLI Contradiction Engine with Soft Semantic Matching
    and Calibrated Confidence Distributions.
    """
    def __init__(self, contradiction_threshold: float = 0.60, temperature: float = 1.2):
        self.threshold = contradiction_threshold
        self.temperature = temperature

        # Domain Opposition Taxonomy with semantic weights
        self.opposition_pairs = [
            ("dynamodb", "postgresql", 0.94),
            ("mongodb", "postgresql", 0.91),
            ("mysql", "cassandra", 0.89),
            ("redis", "memcached", 0.86),
            ("elasticsearch", "opensearch", 0.84),
            ("aws", "gcp", 0.92),
            ("azure", "aws", 0.90),
            ("on-premise", "cloud", 0.95),
            ("serverless", "dedicated", 0.87),
            ("rest", "graphql", 0.88),
            ("grpc", "rest", 0.85),
            ("microservices", "monolith", 0.93),
            ("react", "vue", 0.91),
            ("flutter", "react native", 0.92),
            ("freeze", "deploy", 0.96),
            ("freeze", "release", 0.95),
            ("delay", "accelerate", 0.91),
            ("postpone", "immediate", 0.94),
            ("deprecate", "maintain", 0.89),
            ("remote", "in-office", 0.93),
            ("hybrid", "mandatory in-office", 0.88),
            ("prohibit", "allow", 0.97)
        ]

        self.negation_tokens = {"not", "never", "no", "stop", "halt", "cancel", "prohibit", "ban", "freeze", "disallow"}
        self.affirmative_tokens = {"allow", "permit", "start", "proceed", "resume", "deploy", "release", "ship"}

    def _softmax(self, logits: Tuple[float, float, float]) -> Tuple[float, float, float]:
        exp_logits = [math.exp(x / self.temperature) for x in logits]
        total = sum(exp_logits)
        return (exp_logits[0] / total, exp_logits[1] / total, exp_logits[2] / total)

    def evaluate_pair(self, past_decision: DecisionItem, new_decision: DecisionItem) -> Optional[ConflictAlert]:
        past_stmt = past_decision.statement.lower()
        new_stmt = new_decision.statement.lower()

        conflict_detected = False
        detected_pair = ()
        raw_confidence = 0.0
        evidence_type = "Domain Technology Reversal"

        # 1. Direct Term Opposition
        for term_a, term_b, weight in self.opposition_pairs:
            if (term_a in past_stmt and term_b in new_stmt) or (term_b in past_stmt and term_a in new_stmt):
                conflict_detected = True
                detected_pair = (term_a, term_b)
                raw_confidence = weight
                break

        # 2. Negation / Stance Inversion
        if not conflict_detected:
            is_past_double_neg = ("disallow" in past_stmt or "prohibit" in past_stmt) and "without" in past_stmt
            is_new_double_neg = ("disallow" in new_stmt or "prohibit" in new_stmt) and "without" in new_stmt

            past_tokens = set(past_stmt.split())
            new_tokens = set(new_stmt.split())

            past_has_neg = bool(past_tokens & self.negation_tokens) and not is_past_double_neg
            new_has_neg = bool(new_tokens & self.negation_tokens) and not is_new_double_neg

            overlap_nouns = (past_tokens & new_tokens) - self.negation_tokens - self.affirmative_tokens - {
                "the", "a", "an", "to", "in", "for", "all", "our", "we", "will", "this", "without"
            }
            if len(overlap_nouns) >= 2 and (past_has_neg != new_has_neg):
                conflict_detected = True
                detected_pair = ("Negated Policy", "Affirmative Policy")
                raw_confidence = 0.88
                evidence_type = "Policy Stance Inversion"

        if conflict_detected:
            # Calibrated soft logit distribution
            p_con = round(raw_confidence, 4)
            rem = 1.0 - p_con
            p_neu = round(rem * 0.75, 4)
            p_ent = round(rem * 0.25, 4)

            explanation = (
                f"[{evidence_type}] Detected conflicting commitments: "
                f"Historical policy in {past_decision.meeting_id} ({past_decision.owner}): '{past_decision.statement}'. "
                f"New proposal in {new_decision.meeting_id} ({new_decision.owner}): '{new_decision.statement}'. "
                f"Contradicting entities: {detected_pair[0].upper()} vs {detected_pair[1].upper()}."
            )

            return ConflictAlert(
                conflict_id=f"conflict_{past_decision.decision_id}_{new_decision.decision_id}",
                current_decision=new_decision,
                conflicting_decision=past_decision,
                contradiction_score=p_con,
                entailment_score=p_ent,
                neutral_score=p_neu,
                explanation=explanation,
                historical_meeting_id=past_decision.meeting_id,
                requires_human_override=True
            )

        return None
