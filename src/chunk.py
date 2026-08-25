"""Splits a document into passages, one string per chunk.

Invariant: chunk_text always returns non-empty, stripped strings -- no empty
passages ever enter the graph or the index. A blank passage would sit in the
BM25/embedding index with zero signal and just dilute every ranking.

Paragraphs (blank-line separated) are the natural unit. A paragraph longer
than ~6 sentences gets split further with a sliding window so no single
passage is so long it drowns out everything else in a cosine-similarity
search.
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WINDOW_SENTENCES = 4
_WINDOW_STRIDE = 3


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]


def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        sentences = _split_sentences(para)
        if len(sentences) <= _WINDOW_SENTENCES:
            chunks.append(para)
            continue
        for i in range(0, len(sentences), _WINDOW_STRIDE):
            window = sentences[i:i + _WINDOW_SENTENCES]
            if window:
                chunks.append(" ".join(window))
            if i + _WINDOW_SENTENCES >= len(sentences):
                break
    return chunks
