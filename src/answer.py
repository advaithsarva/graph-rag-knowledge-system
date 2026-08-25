"""Formats a QueryResult into a citable, human-readable answer.

Deliberately extractive, not generative: no LLM call, so no API key and no
hallucination risk. The "credential-free architecture" line in the project
spec is a direct response to the predecessor project (docs-knowledge-graph-qa
/ NvidiaHackathon) requiring a hosted-model credential to run at all. The
tradeoff is real -- this reads like a structured search result, not a fluent
sentence -- and it's the one the spec explicitly asks for.
"""
from src.retrieve import QueryResult


def format_answer(result: QueryResult) -> str:
    lines = []

    if result.relation_chain:
        lines.append("Reasoning path (from the knowledge graph):")
        for subj, pred, obj in result.relation_chain:
            lines.append(f"  {subj} --{pred}--> {obj}")
        lines.append("")

    if not result.results:
        lines.append("No relevant passages found.")
        return "\n".join(lines)

    lines.append("Supporting passages:")
    for r in result.results:
        lines.append(f"  [{r.passage_id}] (score={r.score}, via={'+'.join(r.via)})")
        lines.append(f"    {r.text}")

    return "\n".join(lines)
