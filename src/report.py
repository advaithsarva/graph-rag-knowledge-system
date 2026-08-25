"""Renders a query's traversed subgraph as one self-contained HTML+SVG page.

Plain SVG computed from networkx's spring_layout, not a JS graph library --
a query subgraph here is a few dozen nodes at most, well within what a
static layout handles, and it keeps the report a single file with no
external script to load.
"""
import html
import networkx as nx

_ENTITY_COLOR = "#f4a261"
_PASSAGE_COLOR = "#8ecae6"
_CANVAS = 700
_MARGIN = 60


def _positions(subgraph: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    if subgraph.number_of_nodes() == 0:
        return {}
    raw = nx.spring_layout(subgraph, seed=42)
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = (x_hi - x_lo) or 1
    y_span = (y_hi - y_lo) or 1
    scaled = {}
    for node, (x, y) in raw.items():
        sx = _MARGIN + (x - x_lo) / x_span * (_CANVAS - 2 * _MARGIN)
        sy = _MARGIN + (y - y_lo) / y_span * (_CANVAS - 2 * _MARGIN)
        scaled[node] = (sx, sy)
    return scaled


def render_subgraph(subgraph: nx.MultiDiGraph, title: str = "Query subgraph") -> str:
    pos = _positions(subgraph)
    svg_parts = []

    for u, v, data in subgraph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        label = data.get("predicate", data.get("kind", ""))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        svg_parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#999" stroke-width="1"/>')
        if label and label != "mentions":
            svg_parts.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="9" fill="#666">{html.escape(label)}</text>')

    for node, (x, y) in pos.items():
        data = subgraph.nodes[node]
        is_entity = data.get("kind") == "entity"
        color = _ENTITY_COLOR if is_entity else _PASSAGE_COLOR
        label = data.get("label", node) if is_entity else node
        r = 8 if is_entity else 14
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="#333" stroke-width="1"/>'
        svg_parts.append(shape)
        svg_parts.append(f'<text x="{x:.1f}" y="{y - r - 4:.1f}" font-size="10" text-anchor="middle">{html.escape(str(label))[:24]}</text>')

    svg = f'<svg width="{_CANVAS}" height="{_CANVAS}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_parts)}</svg>'

    legend = (
        f'<p><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{_ENTITY_COLOR}"/></svg> entity &nbsp;'
        f'<svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{_PASSAGE_COLOR}"/></svg> passage</p>'
    )

    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;max-width:800px;margin:2rem auto}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{legend}{svg}</body></html>"
    )
