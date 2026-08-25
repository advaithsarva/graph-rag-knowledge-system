"""Relation extraction tests -- specifically the passive-voice bug found while
building this: the naive version emitted (Acme Corp, acquire, 2020) instead
of (Globex, acquire, Acme Corp) for "Acme Corp was acquired by Globex in
2020." because it treated the grammatical subject as the semantic subject.
Needs spaCy's real parser, so this isn't network-free, but the model is
already local after the first `pip install`/`spacy download`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import extract_relations, extract_entities


def test_active_voice_svo():
    triples = extract_relations("Alice founded Acme Corp in 2010.")
    assert ("Alice", "found", "Acme Corp") in triples


def test_passive_voice_recovers_true_agent():
    triples = extract_relations("Acme Corp was acquired by Globex in 2020.")
    assert ("Globex", "acquire", "Acme Corp") in triples
    # the grammatical subject must NOT appear as the semantic subject
    assert not any(subj == "Acme Corp" for subj, _, _ in triples)


def test_passive_without_agent_is_skipped_not_guessed():
    triples = extract_relations("Acme Corp was founded in 2010.")
    assert triples == []


def test_entities_found():
    ents = extract_entities("Alice founded Acme Corp in 2010.")
    assert any(e["text"] == "2010" and e["label"] == "DATE" for e in ents)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
