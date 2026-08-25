"""Entities and (subject, predicate, object) relation triples from one passage.

Both run on spaCy's dependency parse -- no separately trained relation-
extraction model. The tradeoff is explicit: this catches simple, direct
sentences ("Alice founded Acme Corp.") and misses passive voice, coreference
("It was acquired by...") and anything needing world knowledge. That's the
honest ceiling of a rule-based extractor; see CLAUDE.md for the reasoning
behind not reaching for a bigger model here.
"""
from functools import lru_cache

_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load(_MODEL)


def extract_entities(text: str) -> list[dict]:
    if not text.strip():
        return []
    doc = _nlp()(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


def _span_text(token) -> str:
    """Subject/object phrase: the token plus its compound/modifier children,
    so 'Acme Corp' comes back whole instead of just 'Corp'."""
    span_tokens = sorted([token] + [c for c in token.children if c.dep_ in ("compound", "amod", "poss")],
                          key=lambda t: t.i)
    return " ".join(t.text for t in span_tokens)


def extract_relations(text: str) -> list[tuple[str, str, str]]:
    """(subject, predicate, object) triples, one attempt per verb per sentence.

    Passive voice ("X was acquired by Y") is handled explicitly: spaCy marks
    the "by"-phrase with dep_ == 'agent', and the real actor is that
    preposition's object, not the grammatical subject. Treating nsubjpass as
    the subject without this check silently produces backwards or nonsense
    triples -- e.g. it would emit (Acme, acquire, 2020) instead of
    (Globex, acquire, Acme) for "Acme was acquired by Globex in 2020."
    A passive sentence with no "by"-agent (very common: "X was founded in
    1990") gives no discoverable actor, so it's skipped rather than guessed.
    """
    if not text.strip():
        return []
    doc = _nlp()(text)
    triples = []
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue

            passive_subj = next((c for c in token.children if c.dep_ == "nsubjpass"), None)
            if passive_subj is not None:
                agent_prep = next((c for c in token.children if c.dep_ == "agent"), None)
                agent = next((c for c in agent_prep.children if c.dep_ == "pobj"), None) if agent_prep else None
                if agent is not None:
                    triples.append((_span_text(agent), token.lemma_, _span_text(passive_subj)))
                continue

            subjects = [c for c in token.children if c.dep_ == "nsubj"]
            objects = [c for c in token.children if c.dep_ in ("dobj", "attr")]
            if not objects:
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    objects.extend(c for c in prep.children if c.dep_ == "pobj")
            if subjects and objects:
                triples.append((_span_text(subjects[0]), token.lemma_, _span_text(objects[0])))
    return triples
