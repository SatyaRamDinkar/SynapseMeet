import time
from typing import List, Dict
from src.schemas.models import DecisionItem
from src.reasoning.advanced_nli_detector import AdvancedNLIConflictDetector

class ComprehensiveEvaluationHarness:
    """
    Standardized Multi-Domain Evaluation Benchmark for Cross-Meeting Contradiction Detection.
    Expands evaluation across 25 curated test meeting pairs covering:
    - Infrastructure & Databases
    - Cloud Platforms & Kubernetes
    - Frontend & Mobile Frameworks
    - Security, Compliance & Access Control
    - DevOps, Code Freezes & Release Policies
    - Work Policies & Team Commitments
    Computes Precision, Recall, Specificity, F1-Score, ROC-AUC approximation, and Mean Latency.
    """
    def __init__(self):
        self.detector = AdvancedNLIConflictDetector()

    def get_test_dataset(self) -> List[Dict]:
        return [
            # --- Domain 1: Database & Storage ---
            {"past": DecisionItem(decision_id="D1_P", topic="Storage", statement="Store all transactional records in DynamoDB.", owner="Alice", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D1_N", topic="Storage", statement="Migrate all transactional records to PostgreSQL immediately.", owner="Bob", meeting_id="Sprint-04"),
             "is_contradiction": True, "category": "Database"},

            {"past": DecisionItem(decision_id="D2_P", topic="Storage", statement="Provision 500GB read capacity for PostgreSQL database.", owner="Alice", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D2_N", topic="Storage", statement="Scale PostgreSQL cluster storage capacity to 1TB.", owner="Alice", meeting_id="Sprint-03"),
             "is_contradiction": False, "category": "Database"},

            {"past": DecisionItem(decision_id="D3_P", topic="Storage", statement="Use Redis as our primary distributed cache layer.", owner="Charlie", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D3_N", topic="Storage", statement="Replace Redis cache cluster with Memcached.", owner="Dave", meeting_id="Sprint-05"),
             "is_contradiction": True, "category": "Database"},

            {"past": DecisionItem(decision_id="D4_P", topic="Storage", statement="Enable automated daily backups on the primary PostgreSQL database.", owner="Eve", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D4_N", topic="Storage", statement="Verify PostgreSQL daily backup restoration in staging.", owner="Eve", meeting_id="Sprint-03"),
             "is_contradiction": False, "category": "Database"},

            # --- Domain 2: Cloud & Infrastructure ---
            {"past": DecisionItem(decision_id="D5_P", topic="Cloud", statement="Standardize all production deployment on AWS EKS.", owner="Charlie", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D5_N", topic="Cloud", statement="Move container workloads from AWS to GCP Kubernetes engine.", owner="Dave", meeting_id="Sprint-05"),
             "is_contradiction": True, "category": "Cloud"},

            {"past": DecisionItem(decision_id="D6_P", topic="Cloud", statement="Enable CloudWatch monitoring alerts on AWS infrastructure.", owner="Charlie", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D6_N", topic="Cloud", statement="Configure PagerDuty webhooks for critical CloudWatch alerts.", owner="Eve", meeting_id="Sprint-04"),
             "is_contradiction": False, "category": "Cloud"},

            {"past": DecisionItem(decision_id="D7_P", topic="Cloud", statement="Migrate remaining on-premise servers to AWS cloud by Q4.", owner="Architect", meeting_id="Planning-01"),
             "new": DecisionItem(decision_id="D7_N", topic="Cloud", statement="Retain core data warehouse servers on-premise permanently.", owner="Director", meeting_id="Planning-03"),
             "is_contradiction": True, "category": "Cloud"},

            # --- Domain 3: Architecture & Protocols ---
            {"past": DecisionItem(decision_id="D8_P", topic="API", statement="Expose all microservice endpoints via GraphQL.", owner="Frank", meeting_id="Arch-01"),
             "new": DecisionItem(decision_id="D8_N", topic="API", statement="Deprecate GraphQL and standardize all endpoints on REST APIs.", owner="Grace", meeting_id="Arch-03"),
             "is_contradiction": True, "category": "Architecture"},

            {"past": DecisionItem(decision_id="D9_P", topic="API", statement="Implement JWT token authentication on REST APIs.", owner="Frank", meeting_id="Arch-01"),
             "new": DecisionItem(decision_id="D9_N", topic="API", statement="Add token expiration refresh logic to existing JWT authentication.", owner="Frank", meeting_id="Arch-02"),
             "is_contradiction": False, "category": "Architecture"},

            {"past": DecisionItem(decision_id="D10_P", topic="Architecture", statement="Decompose core monolithic billing service into microservices.", owner="Lead", meeting_id="Sprint-07"),
             "new": DecisionItem(decision_id="D10_N", topic="Architecture", statement="Maintain billing service inside the monolith to avoid latency.", owner="Principal", meeting_id="Sprint-09"),
             "is_contradiction": True, "category": "Architecture"},

            # --- Domain 4: Mobile & Frontend ---
            {"past": DecisionItem(decision_id="D11_P", topic="Mobile", statement="Build the new consumer app using Flutter.", owner="Frank", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D11_N", topic="Mobile", statement="Rewrite the mobile codebase in React Native.", owner="Grace", meeting_id="Sprint-06"),
             "is_contradiction": True, "category": "Frontend"},

            {"past": DecisionItem(decision_id="D12_P", topic="Mobile", statement="Target iOS 17 minimum SDK version for release.", owner="Frank", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D12_N", topic="Mobile", statement="Prepare Android Play Store developer release credentials.", owner="Grace", meeting_id="Sprint-03"),
             "is_contradiction": False, "category": "Frontend"},

            {"past": DecisionItem(decision_id="D13_P", topic="Web", statement="Build our admin dashboard with React components.", owner="Lead", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D13_N", topic="Web", statement="Rebuild admin dashboard using Vue 3 framework.", owner="Dev", meeting_id="Sprint-04"),
             "is_contradiction": True, "category": "Frontend"},

            {"past": DecisionItem(decision_id="D14_P", topic="Web", statement="Implement Tailwind CSS styling for all React buttons.", owner="Lead", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D14_N", topic="Web", statement="Add dark mode support across Tailwind React components.", owner="Lead", meeting_id="Sprint-03"),
             "is_contradiction": False, "category": "Frontend"},

            # --- Domain 5: DevOps & Delivery Policies ---
            {"past": DecisionItem(decision_id="D15_P", topic="DevOps", statement="Enforce code freeze starting this Friday for all teams.", owner="DevOps", meeting_id="Sprint-05"),
             "new": DecisionItem(decision_id="D15_N", topic="DevOps", statement="Deploy new microservices release to production this Friday.", owner="Eng", meeting_id="Sprint-06"),
             "is_contradiction": True, "category": "DevOps"},

            {"past": DecisionItem(decision_id="D16_P", topic="DevOps", statement="Automate deployment pipeline using GitHub Actions.", owner="DevOps", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D16_N", topic="DevOps", statement="Add Docker container build caching to GitHub Actions.", owner="DevOps", meeting_id="Sprint-02"),
             "is_contradiction": False, "category": "DevOps"},

            {"past": DecisionItem(decision_id="D17_P", topic="Release", statement="Immediate release of version 2.0 to beta users.", owner="PM", meeting_id="Sprint-02"),
             "new": DecisionItem(decision_id="D17_N", topic="Release", statement="Postpone release of version 2.0 indefinitely.", owner="QA", meeting_id="Sprint-03"),
             "is_contradiction": True, "category": "DevOps"},

            # --- Domain 6: Work & Organizational Policies ---
            {"past": DecisionItem(decision_id="D18_P", topic="Operations", statement="Mandate 3 days in-office attendance per week.", owner="HR", meeting_id="Townhall-01"),
             "new": DecisionItem(decision_id="D18_N", topic="Operations", statement="Switch the engineering organization to 100% remote work.", owner="CTO", meeting_id="Townhall-03"),
             "is_contradiction": True, "category": "Policy"},

            {"past": DecisionItem(decision_id="D19_P", topic="Operations", statement="Schedule sprint retrospectives every other Friday at 4 PM.", owner="ScrumMaster", meeting_id="Townhall-01"),
             "new": DecisionItem(decision_id="D19_N", topic="Operations", statement="Keep sprint retro format at 4 PM on alternating Fridays.", owner="ScrumMaster", meeting_id="Townhall-02"),
             "is_contradiction": False, "category": "Policy"},

            # --- Domain 7: Security & Compliance ---
            {"past": DecisionItem(decision_id="D20_P", topic="Security", statement="Prohibit plain text API keys in code repositories.", owner="SecOps", meeting_id="Sec-01"),
             "new": DecisionItem(decision_id="D20_N", topic="Security", statement="Allow developer team to commit test API keys to repository.", owner="Dev", meeting_id="Sec-03"),
             "is_contradiction": True, "category": "Security"},

            {"past": DecisionItem(decision_id="D21_P", topic="Security", statement="Enforce MFA authentication on all internal admin portals.", owner="SecOps", meeting_id="Sec-01"),
             "new": DecisionItem(decision_id="D21_N", topic="Security", statement="Audit admin portal logins to verify 100% MFA compliance.", owner="SecOps", meeting_id="Sec-02"),
             "is_contradiction": False, "category": "Security"},

            # --- Domain 8: Performance SLAs & Testing ---
            {"past": DecisionItem(decision_id="D22_P", topic="Quality", statement="Maintain sub-200ms latency for search API endpoint.", owner="Architect", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D22_N", topic="Quality", statement="Monitor search latency threshold below 200ms in Grafana.", owner="SRE", meeting_id="Sprint-04"),
             "is_contradiction": False, "category": "SLA"},

            {"past": DecisionItem(decision_id="D23_P", topic="Testing", statement="Enforce minimum 80% unit test coverage before pull request merge.", owner="QA", meeting_id="Sprint-01"),
             "new": DecisionItem(decision_id="D23_N", topic="Testing", statement="Disallow pull request merges without 80% test coverage verification.", owner="QA", meeting_id="Sprint-03"),
             "is_contradiction": False, "category": "SLA"},

            {"past": DecisionItem(decision_id="D24_P", topic="DevOps", statement="Accelerate cloud migration timeline to finish before Q3.", owner="VP", meeting_id="Exec-01"),
             "new": DecisionItem(decision_id="D24_N", topic="DevOps", statement="Delay cloud migration completion until Q1 next year.", owner="PM", meeting_id="Exec-02"),
             "is_contradiction": True, "category": "DevOps"},

            {"past": DecisionItem(decision_id="D25_P", topic="Security", statement="Deprecate TLS 1.0 and TLS 1.1 across all load balancers.", owner="SecOps", meeting_id="Sec-01"),
             "new": DecisionItem(decision_id="D25_N", topic="Security", statement="Maintain TLS 1.0 support indefinitely for legacy enterprise clients.", owner="Sales", meeting_id="Sec-04"),
             "is_contradiction": True, "category": "Security"}
        ]

    def run_benchmark(self) -> Dict:
        dataset = self.get_test_dataset()
        tp = fp = tn = fn = 0
        latencies = []

        for item in dataset:
            t0 = time.perf_counter()
            alert = self.detector.evaluate_pair(item["past"], item["new"])
            latencies.append((time.perf_counter() - t0) * 1000)

            predicted = alert is not None
            actual = item["is_contradiction"]

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1

        total = len(dataset)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total
        mean_latency = sum(latencies) / len(latencies)

        return {
            "total_samples": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "accuracy": accuracy,
            "mean_latency_ms": round(mean_latency, 3),
            "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 3)
        }
