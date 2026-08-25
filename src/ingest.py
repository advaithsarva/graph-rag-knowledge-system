"""Document ingestion: chunk -> NER -> relation extraction -> graph + index upsert."""
from src.chunk import chunk_text
from src.extract import extract_entities, extract_relations
from src.graph import KnowledgeGraph
from src.index import PassageIndex


def ingest(doc_id: str, text: str, graph: KnowledgeGraph, index: PassageIndex):
    for i, passage in enumerate(chunk_text(text)):
        passage_id = f"{doc_id}#{i}"
        graph.add_passage(
            passage_id, passage, doc_id,
            entities=extract_entities(passage),
            relations=extract_relations(passage),
        )
        index.add(passage_id, passage)


def build_knowledge_base(documents: dict[str, str]) -> tuple[KnowledgeGraph, PassageIndex]:
    """documents: {doc_id: full_text}"""
    graph = KnowledgeGraph()
    index = PassageIndex()
    for doc_id, text in documents.items():
        ingest(doc_id, text, graph, index)
    index.build()
    return graph, index
