"""BM25 keyword index + embedding vector index over the same passages.

Both are built fresh from `build()` rather than updated incrementally --
ingestion in this project is batch ("load these documents"), not a live
stream, so there's no reason to carry the complexity of incremental index
updates. Rebuild is O(passages) and passage counts here are in the hundreds,
not millions.
"""
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class PassageIndex:
    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None

    def add(self, passage_id: str, text: str):
        self.ids.append(passage_id)
        self.texts.append(text)

    def build(self):
        self._bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._embeddings = self._model.encode(self.texts, normalize_embeddings=True)

    def bm25_scores(self, query: str) -> np.ndarray:
        return self._bm25.get_scores(_tokenize(query))

    def vector_scores(self, query: str) -> np.ndarray:
        q_vec = self._model.encode([query], normalize_embeddings=True)[0]
        return self._embeddings @ q_vec  # cosine similarity, both sides unit-normalized

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._top_k(self.bm25_scores(query), k)

    def vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._top_k(self.vector_scores(query), k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]
