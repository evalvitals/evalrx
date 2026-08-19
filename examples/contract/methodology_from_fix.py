"""Turn a real M4 fix candidate into the MethodologyWire the contract wants.

Shows the intended split: a producer describes steps (drawio_builder.MethodGraph,
which is NOT part of the contract), renders them once, and ships only the XML.

Transcribed from ``fixes/08_L2_coded_pipeline`` of the bbh_tracking7 run: 60
lines of generated Python that a reader cannot review, expressed as the eleven
steps it actually performs.

    python examples/contract/methodology_from_fix.py -o coded_pipeline.drawio
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evalvitals.contract.methodology import MethodologyWire
from examples.contract.drawio_builder import Edge, MethodGraph, Node

GRAPH = MethodGraph(
    title="coded_pipeline — vote on content, then re-bind the letter",
    summary=(
        "The model tracks the swaps correctly but prints the wrong option letter. So: solve three "
        "times under different tracking instructions, take a majority vote on the answer TEXT "
        "rather than the letter, then ask the model once more which letter sits next to that exact "
        "text. The letter is looked up, never reasoned about."
    ),
    nodes=[
        Node(id="case", kind="input", label="BBH tracking case\n(prompt + 7 options)"),
        Node(
            id="v1", kind="model_call", touches_model=True, n_calls=1,
            label="V1: full state table\nafter every swap",
            detail="temperature=0.4. Rewrite the COMPLETE holder/item table after each swap; "
                   "only the two named people exchange, everyone else is unchanged.",
        ),
        Node(
            id="v2", kind="model_call", touches_model=True, n_calls=1,
            label="V2: track the asked\nholder step by step",
            detail="temperature=0.7. Identify the target holder first, then restate what THAT "
                   "holder has after every step.",
        ),
        Node(
            id="v3", kind="model_call", touches_model=True, n_calls=1,
            label="V3: solve twice +\npermutation check",
            detail="temperature=0.7. Forward pass, then verify the final table is a valid "
                   "permutation (each item with exactly one holder). Redo on disagreement.",
        ),
        Node(id="parse", kind="transform", label="extract FINAL CONTENT\nand letter from each",
                   detail="Regex on 'FINAL CONTENT:' and 'Answer: (X)'; falls back to \\boxed{}."),
        Node(id="vote", kind="aggregate", label="majority vote\non normalized TEXT",
                   detail="Lowercase, strip punctuation and articles, then Counter.most_common."),
        Node(id="has", kind="decision", label="any text\nagreement?"),
        Node(
            id="bind", kind="model_call", touches_model=True, n_calls=1,
            label="look up the letter\nfor the winning text",
            detail="temperature=0.0. Options list + winning text -> 'Answer: (X)'. Explicitly "
                   "instructed not to solve anything: this step is a lookup, and that is the repair.",
        ),
        Node(id="fb", kind="transform", label="fall back to letter vote\nthen to baseline",
                   detail="No text agreement: vote on letters; if still nothing, return the "
                          "baseline output unchanged."),
        Node(id="ans", kind="output", label="final answer"),
        Node(id="base", kind="baseline", label="unmodified model\n(paired baseline)"),
    ],
    edges=[
        Edge(source="case", target="v1"),
        Edge(source="case", target="v2"),
        Edge(source="case", target="v3"),
        Edge(source="v1", target="parse"),
        Edge(source="v2", target="parse"),
        Edge(source="v3", target="parse"),
        Edge(source="parse", target="vote"),
        Edge(source="vote", target="has"),
        Edge(source="has", target="bind", label="yes"),
        Edge(source="has", target="fb", label="no", kind="fallback"),
        Edge(source="bind", target="ans"),
        Edge(source="fb", target="ans"),
        Edge(source="ans", target="base", label="McNemar paired", kind="compare"),
    ],
)


#: What the contract actually carries: title, summary, and the diagram.
METHODOLOGY = MethodologyWire(
    title=GRAPH.title,
    tier="L2",
    summary=GRAPH.summary,
    drawio_xml=GRAPH.to_drawio(),
    n_model_calls=sum(n.n_calls or 0 for n in GRAPH.nodes),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=Path("coded_pipeline.drawio"))
    ap.add_argument("--mermaid", action="store_true", help="print the mermaid form instead")
    args = ap.parse_args()

    if args.mermaid:
        print(GRAPH.to_mermaid())
        return

    args.out.write_text(METHODOLOGY.drawio_xml)
    print(f"{args.out}  ({args.out.stat().st_size} bytes)")
    print(f"  layers            : {max(GRAPH.rank().values()) + 1}")
    print(f"  model calls / case: {METHODOLOGY.n_model_calls}")
    print(f"  model_call_share  : {GRAPH.model_call_share():.0%}")


if __name__ == "__main__":
    main()
