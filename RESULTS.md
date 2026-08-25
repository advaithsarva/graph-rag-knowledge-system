# Results

## Retrieval quality: BM25-only vs. vector-only vs. hybrid

Command:

```bash
python scripts/measure_retrieval.py
```

Corpus: 7 passages across 7 short documents (3 about a company/acquisition
storyline, 4 unrelated distractors — bike lanes, a frog species, a bakery, a
marathon). 5 hand-written queries with known gold passages; one query is
deliberately paraphrased ("Name the person behind the firm now owned by
Globex" instead of "who founded...") so it shares almost no vocabulary with
its gold passage, and can't be answered by semantic similarity to the
passage either — that's the case the graph signal exists for.

Metric is **Mean Reciprocal Rank at 3** (1/rank of the gold passage, 0 if
outside top 3), not plain recall@3 — with only 7 passages, recall@3 saturates
to 5/5 for all three methods (it only has to beat 4 other passages) and
hides the actual differences. MRR still rewards moving the gold passage from
rank 3 to rank 1, which is what the graph signal does below.

| Method | MRR@3 |
|---|---|
| BM25-only | 0.867 |
| Vector-only | 0.733 |
| **Hybrid (this project)** | **0.900** |

Per-query rank of the gold passage:

| Query | BM25 | Vector | Hybrid |
|---|---|---|---|
| Who founded the company that Globex acquired? | 1 | 3 | 1 |
| What software did Acme Corp build? | 1 | 1 | 1 |
| Where is Globex based? | 1 | 1 | 1 |
| Who leads Globex as chief executive? | 1 | 1 | 1 |
| Name the person behind the firm now owned by Globex. | 3 | 3 | **2** |

Reading these honestly: on straightforward queries all three methods tie —
there's nothing for the graph to add when the answer is already lexically
and semantically obvious. The graph signal earns its keep on exactly two
rows: it recovers rank 1 on the direct multi-hop question that vector-only
drops to rank 3 on, and it's the only method that moves the paraphrased
query's gold passage off last place. Five queries is a demonstration, not a
benchmark — it shows the mechanism works, not a general accuracy claim.

## Latency

5 queries (retrieval only, ingestion excluded) in 0.12s = ~24ms/query on
CPU. Ingesting the 7-document corpus (chunking + spaCy NER/dependency parse
+ embedding all 7 passages) takes a few seconds, dominated by loading the
spaCy and sentence-transformers models on first call — both are cached via
`lru_cache`/module-level singletons after that.

## Multi-hop demonstration (qualitative)

```bash
python -m src.cli samples/doc_*.txt --query "Who founded the company that was acquired by Globex?"
```

Output includes the reasoning path pulled from the graph:

```
Alice --found--> Acme Corp
Globex --acquire--> Acme Corp
```

`doc_company#0` (which never mentions "Globex") is retrieved with
`via=bm25+graph` — direct evidence the graph term, not lexical overlap
alone, is what connects the query to the right passage.

## Test suite

```
tests/test_extract.py    4 passed  (incl. the passive-voice extraction fix)
tests/test_graph.py      4 passed  (multi-hop traversal, offline/no models)
tests/test_retrieve.py   2 passed  (hybrid fusion, real models)
```
