"""Graph traversal: the multi-hop invariant this whole project exists to
demonstrate -- an entity two relation-hops away from the query entity must
be reachable, and it must work walking edges backwards (predecessors), not
just forwards, since "who founded the company Globex acquired" needs to walk
acquire and found both against their natural subject->object direction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import KnowledgeGraph


def _two_hop_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_passage("p1", "Alice founded Acme Corp.", "doc1",
                   entities=[{"text": "Alice", "label": "PERSON", "start": 0, "end": 5},
                             {"text": "Acme Corp", "label": "ORG", "start": 14, "end": 23}],
                   relations=[("Alice", "found", "Acme Corp")])
    g.add_passage("p2", "Acme Corp was acquired by Globex.", "doc2",
                   entities=[{"text": "Acme Corp", "label": "ORG", "start": 0, "end": 9},
                             {"text": "Globex", "label": "ORG", "start": 25, "end": 31}],
                   relations=[("Globex", "acquire", "Acme Corp")])
    return g


def test_one_hop_reaches_direct_neighbor():
    g = _two_hop_graph()
    sub = g.traverse(["globex"], hops=1)
    assert "acme corp" in sub.nodes


def test_two_hops_reaches_alice_from_globex():
    g = _two_hop_graph()
    sub = g.traverse(["globex"], hops=2)
    assert "alice" in sub.nodes, "two hops from Globex (via Acme Corp) must reach Alice"


def test_one_hop_does_not_reach_alice():
    g = _two_hop_graph()
    sub = g.traverse(["globex"], hops=1)
    assert "alice" not in sub.nodes, "Alice is two hops away, one hop must not reach her"


def test_passages_in_subgraph_includes_both_docs():
    g = _two_hop_graph()
    sub = g.traverse(["globex"], hops=2)
    assert g.passages_in(sub) == {"p1", "p2"}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
