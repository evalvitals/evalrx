"""Optional authoring helper: build a draw.io diagram from a step graph.

Not part of the contract. :class:`~evalrx.contract.methodology.MethodologyWire`
takes ``drawio_xml`` and nothing else, so a producer may author that XML any way
it likes. This module is one such way, for producers that would rather describe
steps than place boxes: it computes a layered layout and emits the XML, which
means no hand-invented coordinates and no overlapping nodes.

    from examples.contract.drawio_builder import MethodGraph, Node, Edge
    xml = MethodGraph(title="...", nodes=[...], edges=[...]).to_drawio()
    MethodologyWire(title="...", drawio_xml=xml)

The graph also answers one question the XML cannot: how much of a repair
actually touches the model. ``model_call_share()`` is the numeric form of the
``verdict="model_independent"`` distinction.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Literal

from pydantic import Field, model_validator

from evalrx.contract.common import WireModel

NodeKind = Literal[
    "input",       # the case as it arrives
    "model_call",  # a call to the model under diagnosis
    "transform",   # deterministic computation (parse, normalize, rewrite)
    "decision",    # a branch
    "aggregate",   # merge several paths (vote, pick, concatenate)
    "baseline",    # the unmodified path, for contrast
    "output",      # what is finally returned
]

#: Fill / stroke per node kind. Model calls are the visually loudest thing on the
#: canvas because "how much of this is the model" is the first question a reader
#: has. Legible on both light and dark canvases.
_STYLE: dict[NodeKind, str] = {
    "input":      "rounded=1;fillColor=#EEF2F7;strokeColor=#94A3B8;fontColor=#0F172A;",
    "model_call": "rounded=1;fillColor=#DBEAFE;strokeColor=#2563EB;fontColor=#0F172A;strokeWidth=2;",
    "transform":  "rounded=0;fillColor=#F8FAFC;strokeColor=#94A3B8;fontColor=#0F172A;",
    "decision":   "rhombus;fillColor=#FEF3C7;strokeColor=#D97706;fontColor=#0F172A;",
    "aggregate":  "shape=hexagon;perimeter=hexagonPerimeter2;fillColor=#EDE9FE;strokeColor=#7C3AED;fontColor=#0F172A;",
    "baseline":   "rounded=1;dashed=1;fillColor=#F1F5F9;strokeColor=#64748B;fontColor=#334155;",
    "output":     "rounded=1;fillColor=#DCFCE7;strokeColor=#16A34A;fontColor=#0F172A;strokeWidth=2;",
}

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class Node(WireModel):
    """One step. ``label`` is what a reader sees, so keep it human."""

    id: str = Field(min_length=1, max_length=64)
    kind: NodeKind
    label: str = Field(min_length=1, max_length=120, description="Short imperative phrase.")
    detail: str = Field(
        default="", max_length=2000,
        description="Full text for a side panel: the actual prompt, the code excerpt, the rule.",
    )
    touches_model: bool = Field(
        default=False,
        description="True when this step calls the model under diagnosis. Work done in nodes where "
                    "this is False is the scaffold's own computation, not a repair of the model.",
    )
    n_calls: int | None = Field(default=None, description="Model calls per case, when known.")

    @model_validator(mode="after")
    def _kind_matches_model_flag(self) -> "Node":
        if self.kind == "model_call" and not self.touches_model:
            raise ValueError(f"node {self.id!r}: kind='model_call' but touches_model=False")
        if not _ID_RE.match(self.id):
            raise ValueError(f"node id {self.id!r} must be an XML-safe identifier")
        return self


class Edge(WireModel):
    """A transition. ``label`` carries the branch condition when there is one."""

    source: str
    target: str
    label: str = Field(default="", max_length=60)
    kind: Literal["flow", "fallback", "compare"] = Field(
        default="flow",
        description="'fallback' is the path taken on failure; 'compare' links a step to the "
                    "baseline it is measured against, and is drawn dashed rather than as flow.",
    )


class MethodGraph(WireModel):
    """A repair or intervention, as a graph plus a plain-language summary.

    ``drawio_xml`` is a derived field: call :meth:`render_drawio` and store the
    result. It is carried in the payload rather than generated in the browser so
    a frontend can embed it with no toolchain, and so the exact diagram a reader
    saw is reproducible from the persisted run.
    """

    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(
        default="", max_length=1000,
        description="Two or three sentences a non-specialist can follow. Not the code.",
    )
    tier: str = Field(default="", description="L1..L4 for a fix; empty for an intervention.")
    nodes: list[Node] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _graph_is_well_formed(self) -> "MethodGraph":
        ids = [n.id for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate node ids: {sorted(dupes)}")
        known = set(ids)
        for e in self.edges:
            # A dangling edge renders as an arrow into empty space — the diagram
            # still "works" visually, which is exactly why it must fail here.
            if e.source not in known:
                raise ValueError(f"edge source {e.source!r} is not a node")
            if e.target not in known:
                raise ValueError(f"edge target {e.target!r} is not a node")
        if len(self.nodes) > 1:
            connected = {e.source for e in self.edges} | {e.target for e in self.edges}
            orphans = known - connected
            if orphans:
                raise ValueError(f"unreachable nodes would render as floating boxes: {sorted(orphans)}")
        if not any(n.kind == "output" for n in self.nodes):
            raise ValueError("a methodology must end somewhere: no node of kind 'output'")
        return self

    # -- derived views ---------------------------------------------------

    def model_call_share(self) -> float:
        """Fraction of steps that actually touch the model.

        A low share on a candidate claiming a large gain is the shape of a
        model-independent result: the scaffold, not the model, did the work.
        """
        return sum(1 for n in self.nodes if n.touches_model) / len(self.nodes)

    def rank(self) -> dict[str, int]:
        """Longest-path layer per node. Deterministic, so the diagram is stable
        across regenerations (a diagram that moves every run looks like it
        changed when it did not)."""
        adj: dict[str, list[str]] = defaultdict(list)
        indeg: dict[str, int] = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            if e.kind == "compare":
                continue  # a contrast link is not flow and must not push a layer
            adj[e.source].append(e.target)
            indeg[e.target] += 1
        rank = {n.id: 0 for n in self.nodes}
        queue = deque(sorted(i for i, d in indeg.items() if d == 0))
        seen = 0
        while queue:
            cur = queue.popleft()
            seen += 1
            for nxt in adj[cur]:
                rank[nxt] = max(rank[nxt], rank[cur] + 1)
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if seen != len(self.nodes):
            # A cycle means no layering exists; fall back to declaration order
            # rather than emitting a diagram with silently wrong layers.
            for i, n in enumerate(self.nodes):
                rank[n.id] = i
        return rank

    def to_drawio(self) -> str:
        """Render an uncompressed draw.io (mxfile) document.

        Layout is computed here — layered top-to-bottom, siblings spread across a
        row — so no producer ever hand-writes coordinates.
        """
        W, H, GAP_X, GAP_Y, PAD = 200, 60, 60, 100, 40
        rank = self.rank()
        rows: dict[int, list[Node]] = defaultdict(list)
        for n in self.nodes:
            rows[rank[n.id]].append(n)
        widest = max(len(v) for v in rows.values())
        canvas = widest * W + (widest - 1) * GAP_X

        cells: list[str] = []
        for r in sorted(rows):
            row = rows[r]
            span = len(row) * W + (len(row) - 1) * GAP_X
            x0 = PAD + (canvas - span) / 2
            y = PAD + r * (H + GAP_Y)
            for i, n in enumerate(row):
                x = x0 + i * (W + GAP_X)
                # A literal newline inside an XML attribute is normalized to a
                # space by any conformant parser, so multi-line labels silently
                # collapse. mxGraph renders the entity instead.
                label = html.escape(n.label, quote=True).replace("\n", "&#10;")
                if n.n_calls:
                    label += f"&#10;({n.n_calls}× model)"
                cells.append(
                    f'        <mxCell id="{html.escape(n.id, quote=True)}" value="{label}" '
                    f'style="{_STYLE[n.kind]}whiteSpace=wrap;html=1;align=center;verticalAlign=middle;" '
                    f'vertex="1" parent="1">\n'
                    f'          <mxGeometry x="{x:.0f}" y="{y:.0f}" width="{W}" height="{H}" as="geometry"/>\n'
                    f'        </mxCell>'
                )
        for i, e in enumerate(self.edges):
            dashed = ";dashed=1" if e.kind != "flow" else ""
            color = ";strokeColor=#DC2626" if e.kind == "fallback" else ";strokeColor=#64748B"
            cells.append(
                f'        <mxCell id="edge{i}" value="{html.escape(e.label, quote=True)}" '
                f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;'
                f'endArrow=block;endFill=1{dashed}{color};fontColor=#475569;" '
                f'edge="1" parent="1" source="{html.escape(e.source, quote=True)}" '
                f'target="{html.escape(e.target, quote=True)}">\n'
                f'          <mxGeometry relative="1" as="geometry"/>\n'
                f'        </mxCell>'
            )

        body = "\n".join(cells)
        height = PAD * 2 + (max(rows) + 1) * (H + GAP_Y)
        xml = (
            f'<mxfile host="evalrx" agent="evalrx.contract.methodology" version="21.6.5">\n'
            f'  <diagram id="methodology" name="{html.escape(self.title, quote=True)}">\n'
            f'    <mxGraphModel dx="{canvas + PAD * 2}" dy="{height:.0f}" grid="1" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{canvas + PAD * 2}" pageHeight="{height:.0f}" math="0" shadow="0">\n'
            f'      <root>\n'
            f'        <mxCell id="0"/>\n'
            f'        <mxCell id="1" parent="0"/>\n'
            f'{body}\n'
            f'      </root>\n'
            f'    </mxGraphModel>\n'
            f'  </diagram>\n'
            f'</mxfile>\n'
        )
        return xml

    def to_mermaid(self) -> str:
        """Same graph as mermaid, for readers that prefer text or markdown."""
        shape = {
            "decision": ("{", "}"), "aggregate": ("{{", "}}"),
            "input": ("([", "])"), "output": ("([", "])"),
        }
        lines = ["flowchart TD"]
        for n in self.nodes:
            o, c = shape.get(n.kind, ("[", "]"))
            lines.append(f'    {n.id}{o}"{n.label}"{c}')
        for e in self.edges:
            arrow = "-.->" if e.kind != "flow" else "-->"
            lbl = f'|"{e.label}"|' if e.label else ""
            lines.append(f"    {e.source} {arrow}{lbl} {e.target}")
        return "\n".join(lines)



__all__ = ["Node", "Edge", "MethodGraph"]
