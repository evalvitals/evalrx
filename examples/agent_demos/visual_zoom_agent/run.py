"""Minimal visual tool-loop trajectory — a VLM agent that zooms before answering.

The smallest end-to-end agent-under-test run: Qwen3-VL (hf_local) drives the
backend-agnostic ``Agent`` loop with one visual tool (``image_zoom_in``); the
crop the tool returns is re-injected into the conversation as a new image, and
the whole run is captured as a ``Trajectory`` and written to disk.

Usage (from the repo root, weights already cached):

    .venv/bin/python examples/agent_demos/visual_zoom_agent/run.py --device cuda:0

Outputs land in ``outputs/`` next to this script:
    trajectory.json   the serialized Trajectory (steps, tool calls, metrics)
    case.json         the FailureCase carrying the trajectory
    images/zoom_*.png every crop the tool produced
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image

from evalvitals import Capability, compose
from evalvitals.core.case import FailureCase, Inputs
from evalvitals.models import RuntimeConfig
from evalvitals.models.agent import Agent
from evalvitals.models.tools import zoom_in_tool

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(
    HERE, "..", "..", "m1_m4", "deco_pope", "data", "images",
    "COCO_val2014_000000006033.jpg",
)
DEFAULT_QUESTION = (
    "What is the smallest clearly identifiable object in this image? "
    "Inspect the relevant region closely before answering."
)

SYSTEM = (
    "You are a careful visual reasoning agent. You may call the image_zoom_in "
    "tool to inspect a region of the image at higher resolution; the zoomed "
    "view comes back as a new image. Always zoom into the region most relevant "
    "to the question before answering. When you are confident, give the final "
    "answer as plain text without any tool call."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3-vl-2b-instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    print(f"[1/3] loading {args.model} on {args.device} ...")
    model = compose(
        args.model,
        "hf_local",
        runtime=RuntimeConfig(device=args.device, max_new_tokens=args.max_new_tokens),
        want={Capability.GENERATE, Capability.TOOL_CALLS},
    )

    agent = Agent(
        model,
        tools=[zoom_in_tool(image, save_dir=os.path.join(args.out, "images"))],
        system=SYSTEM,
        max_turns=args.max_turns,
    )
    case = FailureCase(inputs=Inputs(prompt=args.question, image=image))

    print(f"[2/3] running the tool loop (max {args.max_turns} turns) ...")
    trajectory = agent.run(case)
    case.trajectory = trajectory
    case.observed = trajectory.final_answer

    for step in trajectory:
        head = ""
        if step.tool_call:
            head = f"-> {step.tool_call['name']}({step.tool_call['args']})"
        elif step.observation is not None:
            obs = step.observation.get("text") if isinstance(step.observation, dict) else step.observation
            head = f"obs: {str(obs)[:100]}"
        elif step.content:
            head = str(step.content).replace("\n", " ")[:100]
        print(f"  [{step.idx}] {step.role.value:<6} {head}")
    print(f"  metrics: {trajectory.metrics}")
    print(f"  final answer: {trajectory.final_answer!r}")

    traj_path = os.path.join(args.out, "trajectory.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory.to_dict(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "case.json"), "w") as f:
        json.dump(case.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[3/3] wrote {traj_path}")


if __name__ == "__main__":
    main()
