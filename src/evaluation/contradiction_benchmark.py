from typing import List, Dict
from src.schemas.models import DecisionItem
from src.reasoning.nli_conflict_detector import NLIConflictDetector

class ContradictionBenchmark:
    """
    Standardized Quantitative Evaluation Benchmark for Cross-Meeting Contradiction Detection.
    Computes Precision, Recall, F1-Score, and Accuracy across curated test meeting pairs.
    """
    def __init__(self):
        self.detector = NLIConflictDetector()

    def get_test_dataset(self) -> List[Dict]:
        return [
            # 1. Contradiction: Database policy reversal
            {
                "past": DecisionItem(decision_id="T1_P", topic="Storage", statement="Store all transactional records in DynamoDB.", owner="Alice", meeting_id="Sprint-01"),
                "new": DecisionItem(decision_id="T1_N", topic="Storage", statement="Migrate all transactional records to PostgreSQL immediately.", owner="Bob", meeting_id="Sprint-04"),
                "is_contradiction": True
            },
            # 2. Agreement: Storage scaling
            {
                "past": DecisionItem(decision_id="T2_P", topic="Storage", statement="Provision 500GB read capacity for PostgreSQL database.", owner="Alice", meeting_id="Sprint-01"),
                "new": DecisionItem(decision_id="T2_N", topic="Storage", statement="Scale PostgreSQL cluster storage capacity to 1TB.", owner="Alice", meeting_id="Sprint-03"),
                "is_contradiction": False
            },
            # 3. Contradiction: Cloud vendor shift
            {
                "past": DecisionItem(decision_id="T3_P", topic="Cloud", statement="Standardize all production deployment on AWS EKS.", owner="Charlie", meeting_id="Sprint-02"),
                "new": DecisionItem(decision_id="T3_N", topic="Cloud", statement="Move container workloads from AWS to GCP Kubernetes engine.", owner="Dave", meeting_id="Sprint-05"),
                "is_contradiction": True
            },
            # 4. Neutral / Complementary
            {
                "past": DecisionItem(decision_id="T4_P", topic="Cloud", statement="Enable CloudWatch monitoring alerts on AWS.", owner="Charlie", meeting_id="Sprint-02"),
                "new": DecisionItem(decision_id="T4_N", topic="Cloud", statement="Configure PagerDuty webhooks for error alerts.", owner="Eve", meeting_id="Sprint-04"),
                "is_contradiction": False
            },
            # 5. Contradiction: Mobile framework disagreement
            {
                "past": DecisionItem(decision_id="T5_P", topic="Mobile", statement="Build the new consumer app using Flutter.", owner="Frank", meeting_id="Sprint-01"),
                "new": DecisionItem(decision_id="T5_N", topic="Mobile", statement="Rewrite the mobile codebase in React Native.", owner="Grace", meeting_id="Sprint-06"),
                "is_contradiction": True
            },
            # 6. Neutral: Mobile release
            {
                "past": DecisionItem(decision_id="T6_P", topic="Mobile", statement="Target iOS 17 minimum SDK version for release.", owner="Frank", meeting_id="Sprint-02"),
                "new": DecisionItem(decision_id="T6_N", topic="Mobile", statement="Prepare Android Play Store developer release credentials.", owner="Grace", meeting_id="Sprint-03"),
                "is_contradiction": False
            },
            # 7. Contradiction: Work policy
            {
                "past": DecisionItem(decision_id="T7_P", topic="Operations", statement="Mandate 3 days in-office attendance per week.", owner="HR", meeting_id="Townhall-01"),
                "new": DecisionItem(decision_id="T7_N", topic="Operations", statement="Switch the engineering organization to 100% remote work.", owner="CTO", meeting_id="Townhall-03"),
                "is_contradiction": True
            },
            # 8. Contradiction: Code deployment policy
            {
                "past": DecisionItem(decision_id="T8_P", topic="DevOps", statement="Enforce code freeze starting this Friday.", owner="DevOps", meeting_id="Sprint-05"),
                "new": DecisionItem(decision_id="T8_N", topic="DevOps", statement="Deploy new microservices to production this Friday.", owner="Eng", meeting_id="Sprint-06"),
                "is_contradiction": True
            },
            # 9. Agreement: Performance SLA
            {
                "past": DecisionItem(decision_id="T9_P", topic="Quality", statement="Maintain sub-200ms latency for search API endpoint.", owner="Architect", meeting_id="Sprint-01"),
                "new": DecisionItem(decision_id="T9_N", topic="Quality", statement="Monitor search latency threshold below 200ms in Grafana.", owner="SRE", meeting_id="Sprint-04"),
                "is_contradiction": False
            },
            # 10. Contradiction: Release schedule
            {
                "past": DecisionItem(decision_id="T10_P", topic="Release", statement="Immediate release of version 2.0 to beta users.", owner="PM", meeting_id="Sprint-02"),
                "new": DecisionItem(decision_id="T10_N", topic="Release", statement="Postpone release of version 2.0 indefinitely.", owner="QA", meeting_id="Sprint-03"),
                "is_contradiction": True
            }
        ]

    def evaluate(self) -> Dict[str, float]:
        dataset = self.get_test_dataset()
        tp = fp = tn = fn = 0

        for item in dataset:
            alert = self.detector.evaluate_pair(item["past"], item["new"])
            predicted_contradiction = alert is not None
            actual_contradiction = item["is_contradiction"]

            if predicted_contradiction and actual_contradiction:
                tp += 1
            elif predicted_contradiction and not actual_contradiction:
                fp += 1
            elif not predicted_contradiction and not actual_contradiction:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(dataset)

        return {
            "total_samples": len(dataset),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }
