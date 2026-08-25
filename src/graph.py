"""The knowledge graph: passage nodes, entity nodes, and the edges between
them. Backed by networkx instead of a graph database -- no server to run,
no credentials, and networkx's BFS/traversal already covers everything this
project needs. See CLAUDE.md for why ArangoDB (the spec's original choice)
was dropped.

Node kinds, both stored as networkx node attrs (`kind`):
  passage  -- id "{doc_id}#{chunk_index}", attrs: text, doc_id
  entity   -- id is the lowercased entity string (crude but effective
              coreference: "Acme Corp" mentioned twice collapses to one node)

Edge kinds:
  mentions          passage -> entity, added whenever NER finds the entity
                     in that passage
  <predicate>        entity -> entity, added per relation triple, attrs:
                     source_passage (for citing where the claim came from)
"""
import networkx as nx

from src.extract import extract_entities, extract_relations


def _entity_id(text: str) -> str:
    return text.strip().lower()


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_passage(self, passage_id: str, text: str, doc_id: str, entities=None, relations=None):
        self.g.add_node(passage_id, kind="passage", text=text, doc_id=doc_id)

        for ent in entities if entities is not None else extract_entities(text):
            eid = _entity_id(ent["text"])
            self.g.add_node(eid, kind="entity", label=ent["text"], type=ent["label"])
            self.g.add_edge(passage_id, eid, kind="mentions")

        for subj, pred, obj in relations if relations is not None else extract_relations(text):
            s_id, o_id = _entity_id(subj), _entity_id(obj)
            self.g.add_node(s_id, kind="entity", label=self.g.nodes.get(s_id, {}).get("label", subj), type="MISC")
            self.g.add_node(o_id, kind="entity", label=self.g.nodes.get(o_id, {}).get("label", obj), type="MISC")
            self.g.add_edge(s_id, o_id, kind=pred, predicate=pred, source_passage=passage_id)

    def match_entities(self, query: str) -> list[str]:
        """Entity node ids whose label appears (case-insensitive) in the query.
        Simple substring matching, not NER on the query -- the query is
        usually a short question and NER models are tuned for prose, not
        questions ("What company did X found?" often mis-tags 'What company').
        """
        q = query.lower()
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == "entity" and d["label"].lower() in q]

    def traverse(self, start_ids: list[str], hops: int = 2) -> nx.MultiDiGraph:
        """Undirected BFS from the start entities (both find(founded, X->Y) and
        find(Y<-founded, X) queries need to walk edges backwards), returns
        the induced subgraph of everything reached within `hops` steps.
        """
        seen = set(start_ids)
        frontier = set(start_ids)
        for _ in range(hops):
            nxt = set()
            for node in frontier:
                if node not in self.g:
                    continue
                nxt.update(self.g.successors(node))
                nxt.update(self.g.predecessors(node))
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
            if not frontier:
                break
        return self.g.subgraph(seen)

    def passages_in(self, subgraph: nx.MultiDiGraph) -> set[str]:
        return {n for n, d in subgraph.nodes(data=True) if d.get("kind") == "passage"}

    def relation_chain(self, subgraph: nx.MultiDiGraph) -> list[tuple[str, str, str]]:
        """Human-readable (subject_label, predicate, object_label) triples
        found in the traversed subgraph, for showing the reasoning path."""
        chain = []
        for u, v, data in subgraph.edges(data=True):
            if data.get("kind") == "mentions":
                continue
            u_label = self.g.nodes[u].get("label", u)
            v_label = self.g.nodes[v].get("label", v)
            chain.append((u_label, data["predicate"], v_label))
        return chain
