"""Render a vis-network ``{nodes, edges}`` payload as a self-contained HTML.

The bundled vis-network 9.1.2 distribution lives at
``src/extensions/lib/vis-9.1.2/vis-network.min.js`` and ``.css``. We
inline both so the page works inside Streamlit's sandboxed iframe with
no external CDN call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LIB_DIR = (
    Path(__file__).resolve().parent.parent
    / "extensions" / "lib" / "vis-9.1.2"
)


def _read_asset(name: str) -> str:
    p = _LIB_DIR / name
    if not p.is_file():
        raise FileNotFoundError(
            f"Expected vis-network asset at {p} — did you delete the "
            "bundled distribution under src/extensions/lib/vis-9.1.2/ ?"
        )
    return p.read_text(encoding="utf-8")


_PHYSICS_PRESETS: dict[str, dict[str, Any]] = {
    "barnesHut": {
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -8000,
            "centralGravity": 0.3,
            "springLength": 120,
            "springConstant": 0.04,
            "damping": 0.4,
            "avoidOverlap": 0.6,
        },
        "stabilization": {"enabled": True, "iterations": 200, "fit": True},
    },
    "forceAtlas2Based": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08,
            "damping": 0.4,
            "avoidOverlap": 0.5,
        },
        "stabilization": {"enabled": True, "iterations": 200, "fit": True},
    },
    "repulsion": {
        "solver": "repulsion",
        "repulsion": {
            "centralGravity": 0.2,
            "springLength": 200,
            "springConstant": 0.05,
            "nodeDistance": 120,
            "damping": 0.4,
        },
        "stabilization": {"enabled": True, "iterations": 200, "fit": True},
    },
}


def render_vis_html(
    payload: dict[str, Any],
    *,
    height: str = "720px",
    physics: str = "barnesHut",
    background: str = "#ffffff",
    show_legend: bool = True,
) -> str:
    """Render a self-contained HTML page for the given vis-network payload.

    Parameters
    ----------
    payload:
        Output of :func:`fetch_vis_payload`, i.e. ``{"nodes", "edges", "stats"}``.
    height:
        Outer container height (CSS string), e.g. ``"720px"``.
    physics:
        Layout solver — one of ``"barnesHut"``, ``"forceAtlas2Based"``,
        ``"repulsion"``.
    background:
        Canvas background colour.
    show_legend:
        Render a small in-canvas legend in the corner.
    """
    if physics not in _PHYSICS_PRESETS:
        raise ValueError(
            f"physics must be one of {sorted(_PHYSICS_PRESETS)}, got {physics!r}"
        )
    js = _read_asset("vis-network.min.js")
    css = _read_asset("vis-network.css")

    options = {
        "autoResize": True,
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": True,
            "multiselect": True,
        },
        "nodes": {
            "scaling": {"min": 8, "max": 60},
            "shape": "dot",
            "borderWidth": 1,
            "shadow": False,
        },
        "edges": {
            "scaling": {"min": 1, "max": 6},
            "smooth": {"type": "continuous"},
            "shadow": False,
        },
        "physics": _PHYSICS_PRESETS[physics],
        "layout": {"improvedLayout": True},
    }

    legend_html = _legend_html(payload) if show_legend else ""
    payload_json = json.dumps(
        {"nodes": payload["nodes"], "edges": payload["edges"]},
        ensure_ascii=False,
    )
    options_json = json.dumps(options, ensure_ascii=False)

    return _PAGE_TEMPLATE.format(
        css=css,
        js=js,
        background=background,
        height=height,
        payload=payload_json,
        options=options_json,
        legend=legend_html,
    )


def _legend_html(payload: dict[str, Any]) -> str:
    stats = payload.get("stats", {}) or {}
    items = [
        ("Nodes",          str(stats.get("nodes", 0))),
        ("Edges",          str(stats.get("edges", 0))),
        ("Rank metric",    str(stats.get("rank_property", ""))),
        ("Min edge wt.",   f"{stats.get('min_edge_weight', 0)}"),
        ("Coloured by",    str(stats.get("colour_by", ""))),
    ]
    if stats.get("slice_id"):
        items.append(("Slice", str(stats["slice_id"])))
    rows = "".join(
        f"<div class='ln-row'><span>{k}</span><b>{v}</b></div>"
        for k, v in items
    )
    return f"<div class='ln'>{rows}</div>"


_PAGE_TEMPLATE = """\
<!doctype html>
<html><head><meta charset='utf-8'>
<style>{css}</style>
<style>
  html, body {{ margin: 0; padding: 0; background: {background}; }}
  #viz {{ width: 100%; height: {height}; background: {background}; }}
  /* vis-network sets tooltips via innerText (not HTML) and ships nowrap — allow wrapped plain-text titles */
  .vis-tooltip {{
    white-space: pre-wrap !important;
    max-width: min(360px, 92vw);
    font-family: Inter, system-ui, sans-serif;
    font-size: 12px;
    line-height: 1.35;
  }}
  .ln {{
    position: absolute; top: 12px; left: 12px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 6px; padding: 8px 10px;
    font-family: Inter, system-ui, sans-serif; font-size: 12px;
    color: #222; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    z-index: 10;
  }}
  .ln-row {{ display: flex; gap: 12px; justify-content: space-between; }}
  .ln-row b {{ color: #111; }}
</style>
<script>{js}</script>
</head>
<body>
  {legend}
  <div id='viz'></div>
  <script>
    (function() {{
      const data = {payload};
      const options = {options};
      const container = document.getElementById('viz');
      const network = new vis.Network(container, {{
        nodes: new vis.DataSet(data.nodes),
        edges: new vis.DataSet(data.edges)
      }}, options);
      network.once('stabilizationIterationsDone', () => {{
        network.setOptions({{ physics: false }});
        network.fit();
      }});
    }})();
  </script>
</body></html>
"""
