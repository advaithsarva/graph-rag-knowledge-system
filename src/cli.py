"""CLI: ingest a folder of .txt documents, then answer one question.

    python -m src.cli samples/*.txt --query "Who founded the company acquired by Globex?"
"""
import argparse
from pathlib import Path

from src.ingest import build_knowledge_base
from src.retrieve import query as run_query
from src.answer import format_answer
from src.report import render_subgraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--graph-html", type=Path, help="also write a subgraph visualization")
    args = parser.parse_args()

    documents = {f.stem: f.read_text(encoding="utf-8") for f in args.files}
    graph, index = build_knowledge_base(documents)
    print(f"ingested {len(documents)} document(s), {len(index.ids)} passage(s), "
          f"{sum(1 for _, d in graph.g.nodes(data=True) if d['kind'] == 'entity')} entities")

    result = run_query(args.query, graph, index, k=args.k, hops=args.hops)
    print()
    print(format_answer(result))

    if args.graph_html:
        matched = graph.match_entities(args.query)
        subgraph = graph.traverse(matched, hops=args.hops)
        args.graph_html.write_text(render_subgraph(subgraph, title=args.query), encoding="utf-8")
        print(f"\nwrote {args.graph_html}")


if __name__ == "__main__":
    main()
