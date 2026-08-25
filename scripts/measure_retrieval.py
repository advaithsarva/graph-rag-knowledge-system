"""Produces the retrieval comparison numbers in RESULTS.md: rank of the gold
passage (and MRR) for BM25-only, vector-only, and the hybrid fusion this
project ships, over a small hand-written query set with known gold passages.

Mean Reciprocal Rank instead of plain recall@k: with a 7-passage corpus,
recall@3 is nearly saturated for every method (it only has to beat 4 other
passages) and hides the actual differences. MRR still rewards a method for
getting the gold passage from rank 3 to rank 1, which is exactly what the
graph signal does in the "paraphrased" query below.

    python scripts/measure_retrieval.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import build_knowledge_base
from src.retrieve import query as hybrid_query

DOCS_DIR = Path(__file__).resolve().parents[1] / "samples"
K = 3

GOLD = [
    ("Who founded the company that Globex acquired?", "doc_company#0"),
    ("What software did Acme Corp build?", "doc_company#0"),
    ("Where is Globex based?", "doc_acquisition#0"),
    ("Who leads Globex as chief executive?", "doc_ceo#0"),
    # paraphrased to avoid "founded"/"company" so lexical/semantic search
    # can't shortcut it -- this is the case the graph signal exists for
    ("Name the person behind the firm now owned by Globex.", "doc_company#0"),
]


def rank_of(ranked_ids: list[str], gold_id: str) -> int | None:
    """1-indexed rank, or None if not in the top K at all."""
    return ranked_ids.index(gold_id) + 1 if gold_id in ranked_ids else None


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks)


def main():
    documents = {f.stem: f.read_text(encoding="utf-8") for f in DOCS_DIR.glob("doc_*.txt")}
    graph, index = build_knowledge_base(documents)
    print(f"corpus: {len(documents)} docs, {len(index.ids)} passages\n")

    bm25_ranks, vector_ranks, hybrid_ranks = [], [], []
    start = time.perf_counter()
    for question, gold_id in GOLD:
        bm25_ranked = [pid for pid, _ in index.bm25_search(question, K)]
        vector_ranked = [pid for pid, _ in index.vector_search(question, K)]
        hybrid_ranked = [r.passage_id for r in hybrid_query(question, graph, index, k=K).results]

        b, v, h = rank_of(bm25_ranked, gold_id), rank_of(vector_ranked, gold_id), rank_of(hybrid_ranked, gold_id)
        bm25_ranks.append(b); vector_ranks.append(v); hybrid_ranks.append(h)
        print(f"{question}\n  gold={gold_id}  rank: bm25={b} vector={v} hybrid={h}")
    elapsed = time.perf_counter() - start

    print(f"\nMRR@{K}: bm25={mrr(bm25_ranks):.3f}  vector={mrr(vector_ranks):.3f}  hybrid={mrr(hybrid_ranks):.3f}")
    n = len(GOLD)
    print(f"{n} queries in {elapsed:.2f}s ({elapsed/n*1000:.0f}ms/query, ingestion excluded)")


if __name__ == "__main__":
    main()
