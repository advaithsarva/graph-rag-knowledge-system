"""Hybrid retrieval: BM25 + vector similarity + graph traversal, fused into
one ranked list of passages with citations.

Fusion is min-max normalize each signal to [0, 1], then a weighted sum. Not
learned, not tuned on a held-out set -- it's a documented starting point
(see RESULTS.md for the comparison against BM25-only and vector-only), not a
claim that these weights are optimal.
"""
from dataclasses import dataclass, field

import numpy as np

from src.graph import KnowledgeGraph
from src.index import PassageIndex

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.4
GRAPH_WEIGHT = 0.2


def _min_max(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


@dataclass
class RetrievalResult:
    passage_id: str
    doc_id: str
    text: str
    score: float
    via: list[str] = field(default_factory=list)  # which signals contributed


@dataclass
class QueryResult:
    results: list[RetrievalResult]
    matched_entities: list[str]
    relation_chain: list[tuple[str, str, str]]


def query(question: str, graph: KnowledgeGraph, index: PassageIndex, k: int = 5, hops: int = 2) -> QueryResult:
    bm25 = _min_max(index.bm25_scores(question))
    vector = _min_max(index.vector_scores(question))

    matched_entities = graph.match_entities(question)
    subgraph = graph.traverse(matched_entities, hops=hops) if matched_entities else graph.g.subgraph([])
    graph_hit_passages = graph.passages_in(subgraph)

    fused = bm25 * BM25_WEIGHT + vector * VECTOR_WEIGHT
    for i, pid in enumerate(index.ids):
        if pid in graph_hit_passages:
            fused[i] += GRAPH_WEIGHT

    order = np.argsort(fused)[::-1][:k]
    results = []
    for i in order:
        pid = index.ids[i]
        via = []
        if bm25[i] > 0:
            via.append("bm25")
        if vector[i] > 0:
            via.append("vector")
        if pid in graph_hit_passages:
            via.append("graph")
        results.append(RetrievalResult(
            passage_id=pid,
            doc_id=graph.g.nodes[pid]["doc_id"],
            text=graph.g.nodes[pid]["text"],
            score=round(float(fused[i]), 4),
            via=via,
        ))

    return QueryResult(
        results=results,
        matched_entities=[graph.g.nodes[e]["label"] for e in matched_entities],
        relation_chain=graph.relation_chain(subgraph),
    )
