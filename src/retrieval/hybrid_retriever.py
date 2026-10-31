import math
from collections import Counter
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """
    Implements a Hybrid Dense-Sparse RAG retriever combining:
    1. BM25 / Sparse Lexical Matching (exact ticket IDs, version numbers, database technologies).
    2. Dense Semantic Cosine Similarity (conceptual alignment and phrasing).
    Fused via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.documents: List[Dict] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = None

    def index_decisions(self, decisions: List[Dict]):
        self.documents = decisions
        if not decisions:
            return
        corpus = [f"{d['topic']} {d['statement']} {d.get('owner', '')}" for d in decisions]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _bm25_score(self, query: str, doc_text: str) -> float:
        # Fast BM25 approximation
        q_tokens = query.lower().split()
        d_tokens = doc_text.lower().split()
        score = 0.0
        d_len = len(d_tokens) + 1e-5
        for token in q_tokens:
            count = d_tokens.count(token)
            if count > 0:
                tf = count / d_len
                score += (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * (d_len / 30.0)))
        return score

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        if not self.documents or self.tfidf_matrix is None:
            return []

        # 1. Sparse Scores
        sparse_scores = []
        for idx, doc in enumerate(self.documents):
            text = f"{doc['topic']} {doc['statement']}"
            score = self._bm25_score(query, text)
            sparse_scores.append((idx, score))
        sparse_ranked = sorted(sparse_scores, key=lambda x: x[1], reverse=True)

        # 2. Dense / TF-IDF Semantic Scores
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        dense_ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = Counter()
        for rank, (idx, _) in enumerate(sparse_ranked):
            rrf_scores[idx] += 1.0 / (self.k_rrf + rank + 1)
        for rank, (idx, _) in enumerate(dense_ranked):
            rrf_scores[idx] += 1.0 / (self.k_rrf + rank + 1)

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[idx], score) for idx, score in sorted_indices]
