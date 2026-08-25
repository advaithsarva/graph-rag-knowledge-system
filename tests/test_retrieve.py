"""Proves the point of hybrid retrieval: a passage that shares almost no
words with the query, and isn't semantically close to it either, still gets
retrieved because it's connected through the graph -- something a pure
BM25 or pure vector search cannot do. Needs the real embedding model and
spaCy, so this is slower than the other test files (~10-20s once models
are cached locally).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import build_knowledge_base
from src.retrieve import query


def test_graph_signal_surfaces_a_lexically_unrelated_passage():
    docs = {
        "founding": "Alice founded Acme Corp in 2010. Acme Corp built accounting software.",
        "acquisition": "Acme Corp was acquired by Globex in 2020. Globex is based in Chicago.",
    }
    graph, index = build_knowledge_base(docs)

    result = query("Who founded the company that Globex acquired?", graph, index, k=5, hops=2)

    ids = [r.passage_id for r in result.results]
    assert "founding#0" in ids, "the founding passage must be surfaced via the graph hop"

    founding_result = next(r for r in result.results if r.passage_id == "founding#0")
    assert "graph" in founding_result.via

    assert ("Alice", "found", "Acme Corp") in result.relation_chain


def test_bm25_only_would_have_missed_it():
    """Sanity check that the test above is actually testing something: without
    the graph term, the founding passage should rank lower or not appear,
    since it shares almost no vocabulary with the query."""
    docs = {
        "founding": "Alice founded Acme Corp in 2010. Acme Corp built accounting software.",
        "acquisition": "Acme Corp was acquired by Globex in 2020. Globex is based in Chicago.",
    }
    graph, index = build_knowledge_base(docs)
    bm25_scores = index.bm25_scores("Who founded the company that Globex acquired?")
    founding_idx = index.ids.index("founding#0")
    acquisition_idx = index.ids.index("acquisition#0")
    assert bm25_scores[founding_idx] <= bm25_scores[acquisition_idx], (
        "if BM25 alone already favored the founding passage, the graph signal "
        "in the test above wouldn't be proving anything"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
