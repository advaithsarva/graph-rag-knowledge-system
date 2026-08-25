# graph-rag-knowledge-system

A retrieval-augmented generation system on a knowledge-graph backbone.
Ingest documents, extract entities and relations into a graph, then answer
questions with **hybrid retrieval** — BM25 keyword search + vector
similarity + graph traversal, fused into one ranked, cited result. The graph
signal answers multi-hop questions ("who founded the company that Globex
acquired?") that flat retrieval can't, because the answer isn't in any one
passage — it's two relations apart.

Direct evolution of [docs-knowledge-graph-qa](https://github.com/advaithsarva/docs-knowledge-graph-qa)
(formerly NvidiaHackathon), rebuilt with a credential-free architecture.

## Architecture

```
documents --chunk--> passages --spaCy NER + dependency parse--> entities, (subj, pred, obj) triples
                                        |
                                        v
                              networkx knowledge graph
                              (passage nodes, entity nodes,
                               "mentions" and relation edges)

query --> BM25(passages) ┐
      --> cosine sim(embeddings) ├--> min-max normalize, weighted fusion --> ranked, cited passages
      --> graph traversal(matched entities, 2 hops) ┘
```

## Why not ArangoDB / FAISS

The spec calls for ArangoDB + FAISS. This uses **networkx** for the graph
and **sentence-transformers + numpy** for vectors instead — both already
does everything a project this size needs (hundreds of passages, not
millions), and neither needs a server process or a database credential to
run. See [CLAUDE.md](CLAUDE.md) for the reasoning if you're deciding whether
to swap them in later.

## Credential-free, on purpose

No API key, anywhere. Entity/relation extraction is spaCy's dependency
parser (local model), embeddings are a local sentence-transformers model,
and the "answer" is extractive — cited passages plus the graph's reasoning
path, not an LLM-generated sentence. This directly fixes the predecessor
project's core issue (a hosted-model credential required just to run a demo).

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python -m src.cli samples/doc_*.txt \
  --query "Who founded the company that was acquired by Globex?" \
  --graph-html subgraph.html
```

First run downloads spaCy's English model and the ~90MB embedding model;
both are cached locally after.

## Honest limitations

- **Relation extraction is a dependency-parse heuristic, not a trained RE
  model.** It catches direct sentences (active and passive voice) and
  misses coreference ("It was acquired by...") and anything needing world
  knowledge. See `src/extract.py` and the passive-voice bug documented in
  [CLAUDE.md](CLAUDE.md) — an earlier version of this heuristic silently
  produced backwards triples on passive sentences.
- **Entity linking across documents is string matching** (lowercased exact
  match), not real coreference resolution — "Globex" and "Globex Inc." would
  become two separate graph nodes.
- **Fusion weights (0.4 BM25 / 0.4 vector / 0.2 graph) are a documented
  starting point, not a tuned result.** See RESULTS.md.

## Tests

```bash
python tests/test_extract.py    # relation extraction incl. the passive-voice fix
python tests/test_graph.py      # multi-hop traversal invariant, offline
python tests/test_retrieve.py   # hybrid retrieval surfaces what single signals miss
```

## Results

See [RESULTS.md](RESULTS.md): MRR comparison of BM25-only vs. vector-only vs.
hybrid on a 5-query gold set, with the exact command that produced it.

## Project origin

One of three portfolio projects specced in `domains/nlp/`, built the same
day as [multilingual-sentiment-pipeline](https://github.com/advaithsarva/multilingual-sentiment-pipeline)
and [fact-checking-agent](https://github.com/advaithsarva/fact-checking-agent).
