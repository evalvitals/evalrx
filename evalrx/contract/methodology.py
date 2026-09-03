"""Methodology — what an M4 repair or intervention actually DOES.

A candidate's method currently survives only as a code blob or a prompt string.
Neither answers the question a reader has ("what does this do to my model?"), so
the contract carries a diagram instead.

The contract is the draw.io XML and nothing else: a frontend embeds
``drawio_xml`` directly, with no toolchain and no intermediate format to
interpret. How a producer arrives at that XML is its own business —
``examples/contract/drawio_builder.py`` offers a step-graph with automatic
layout for producers that want one.

The XML is still validated on its own terms. A diagram whose edge points at a
cell that does not exist renders as a perfectly ordinary picture, so nothing
downstream — and no reader — would ever notice; that check has to happen here or
not at all.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pydantic import Field, model_validator

from evalrx.contract.common import WireModel


class MethodologyWire(WireModel):
    """A repair or intervention, as an embeddable diagram plus plain language."""

    title: str = Field(min_length=1, max_length=120)
    summary: str = Field(
        default="", max_length=1000,
        description="Two or three sentences a non-specialist can follow. Not the code.",
    )
    tier: str = Field(default="", description="L1..L4 for a fix; empty for an intervention.")
    drawio_xml: str = Field(
        min_length=1,
        description="Uncompressed mxfile document, embeddable as-is. Compressed diagrams are "
                    "rejected: they cannot be diffed, reviewed, or repaired by hand.",
    )
    n_model_calls: int | None = Field(
        default=None,
        description="Model calls this method makes per case. A large claimed gain alongside few "
                    "model calls is the shape of a model-independent result, where the scaffold "
                    "rather than the model did the work.",
    )

    @model_validator(mode="after")
    def _xml_is_renderable(self) -> "MethodologyWire":
        try:
            root = ET.fromstring(self.drawio_xml)
        except ET.ParseError as exc:
            raise ValueError(f"drawio_xml is not well-formed XML: {exc}") from exc
        if root.tag != "mxfile":
            raise ValueError(f"drawio_xml root must be <mxfile>, got <{root.tag}>")
        cells = list(root.iter("mxCell"))
        if not cells:
            raise ValueError("drawio_xml contains no mxCell elements")
        ids = [c.get("id") for c in cells if c.get("id")]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate mxCell ids render unpredictably: {sorted(dupes)}")
        known = set(ids)
        for c in cells:
            for end in ("source", "target"):
                ref = c.get(end)
                if ref and ref not in known:
                    raise ValueError(
                        f"edge {c.get('id')!r} has {end}={ref!r}, which is not a cell in this "
                        "diagram — it draws as an arrow into empty space"
                    )
        return self


__all__ = ["MethodologyWire"]
