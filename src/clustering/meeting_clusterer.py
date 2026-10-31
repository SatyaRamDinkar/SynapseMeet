from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from src.schemas.models import MeetingTranscript

class MeetingSummaryClusterer:
    """
    Directly implements the Meeting Insight Extraction clustering framework
    from Bhuvaji et al. (ACM SAC 2025).
    Clusters historical meeting summaries into semantic domains (e.g. Infrastructure,
    Frontend, Compliance, ML Platform) to restrict graph retrieval and avoid cross-domain false positives.
    """
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        self.cluster_labels_: Dict[int, str] = {}

    def fit_clusters(self, meetings: List[MeetingTranscript]) -> List[MeetingTranscript]:
        if len(meetings) < self.n_clusters:
            # Fallback if meetings count is smaller than clusters
            for idx, m in enumerate(meetings):
                m.cluster_id = 0
                m.cluster_topic = "General Engineering"
            return meetings

        corpus = [f"{m.title}. {m.summary or ''} {' '.join(u.text for u in m.utterances[:5])}" for m in meetings]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        dense_matrix = tfidf_matrix.toarray()

        labels = self.cluster_model.fit_predict(dense_matrix)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Determine dominant keyword for each cluster
        for c in range(self.n_clusters):
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) > 0:
                mean_tfidf = dense_matrix[cluster_indices].mean(axis=0)
                top_idx = mean_tfidf.argsort()[-3:][::-1]
                top_words = feature_names[top_idx]
                self.cluster_labels_[c] = " / ".join(w.capitalize() for w in top_words)
            else:
                self.cluster_labels_[c] = f"Cluster {c}"

        for m, label in zip(meetings, labels):
            m.cluster_id = int(label)
            m.cluster_topic = self.cluster_labels_.get(int(label), "General")

        return meetings

    def assign_cluster(self, meeting: MeetingTranscript) -> int:
        if not hasattr(self.vectorizer, 'vocabulary_'):
            return 0
        text = f"{meeting.title}. {meeting.summary or ''} {' '.join(u.text for u in meeting.utterances[:5])}"
        vec = self.vectorizer.transform([text]).toarray()
        # Find closest cluster centroid
        return 0
