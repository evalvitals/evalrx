"""Hallucination analysis example — POPE + CHAIR.

Demonstrates two black-box hallucination analyzers for VLMs:

  POPE  — yes/no object-presence probes; reports accuracy / F1.
          Paper: Li et al. EMNLP 2023 — https://arxiv.org/abs/2305.10355
          Code:  https://github.com/AoiDragon/POPE

  CHAIR — caption hallucination rate vs gold object vocabulary.
          Paper: Rohrbach et al. EMNLP 2018 — https://arxiv.org/abs/1809.02156

Both analyzers require only GENERATE capability, so any API-backed VLM works.

Usage (inside Docker):
    python run.py                         # uses config.yaml
    python run.py --model gemini-2.0-flash

Expected output:
    [POPE]  accuracy=0.83, f1=0.86, yes_rate=0.50, unparsed=0
    [CHAIR] chair_i=0.25 (mean per-caption hallucination rate), chair_s=0.67
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from PIL import Image

from evalrx.analyzers.hallucination.chair import CHAIRAnalyzer
from evalrx.analyzers.hallucination.pope import POPEAnalyzer
from evalrx.core.case import CaseBatch, FailureCase as Case, Inputs

CONFIG = Path(__file__).parent / "config.yaml"

# Minimal COCO-80 subset for the demo vocabulary
COCO_VOCAB_DEMO = [
    "cat", "dog", "car", "chair", "table", "person", "bird",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
]


def _build_api_model(model_name: str):
    """Construct an API model backed by Gemini (requires GEMINI_API_KEY)."""
    import openai

    from evalrx.models.backends.api import APIModel
    from evalrx.models.backends.base import RuntimeConfig

    client = openai.OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    def generate_fn(prompt, *, model=model_name, image=None, **_):
        content = []
        if image is not None:
            import base64, io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=64,
        )
        return resp.choices[0].message.content or ""

    rt = RuntimeConfig(generate_fn=generate_fn)
    from evalrx.core.spec import ModelSpec
    spec = ModelSpec(key=model_name, family="gemini", model_type="api")
    return APIModel(spec, rt)


def _dummy_image() -> Image.Image:
    """Return a small blank image for the demo (replace with real images)."""
    return Image.new("RGB", (64, 64), color=(200, 200, 200))


def run_pope(model, cfg: dict) -> None:
    """Run POPE on synthetic yes/no probes and print findings."""
    # In production: load real (image, question, gold_label) triples from the
    # POPE dataset (https://github.com/AoiDragon/POPE).
    cases = []
    image = _dummy_image()
    for i in range(cfg.get("pope_n_yes", 3)):
        cases.append(Case(
            id=f"pope_yes_{i}",
            inputs=Inputs(prompt="Is there a cat in the image? Answer yes or no.", image=image),
            metadata={"pope_label": "yes"},
        ))
    for i in range(cfg.get("pope_n_no", 3)):
        cases.append(Case(
            id=f"pope_no_{i}",
            inputs=Inputs(prompt="Is there an airplane in the image? Answer yes or no.", image=image),
            metadata={"pope_label": "no"},
        ))

    result = POPEAnalyzer().run(model, CaseBatch(cases))
    print("[POPE]", result.summary())
    print("  findings:", result.findings)


def run_chair(model, cfg: dict) -> None:
    """Run CHAIR on synthetic caption tasks and print findings."""
    # In production: load (image, gt_objects) pairs from COCO annotations.
    image = _dummy_image()
    cases = [
        Case(
            id=f"chair_{i}",
            inputs=Inputs(prompt="Describe the objects you see in the image.", image=image),
            metadata={"gt_objects": ["cat", "chair"]},
        )
        for i in range(cfg.get("chair_n_cases", 3))
    ]

    result = CHAIRAnalyzer(object_vocab=COCO_VOCAB_DEMO).run(model, CaseBatch(cases))
    print("[CHAIR]", result.summary())
    print("  findings:", result.findings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = args.model or cfg.get("model_name", "gpt-4o-mini")

    model = _build_api_model(model_name)
    run_pope(model, cfg)
    run_chair(model, cfg)


if __name__ == "__main__":
    main()
