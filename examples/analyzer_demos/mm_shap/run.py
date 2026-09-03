"""MM-SHAP + VL-SHAP attribution example.

Demonstrates two Shapley-value-based attribution analyzers for VLMs:

  MM-SHAP — decomposes model reliance between text modality and image modality.
             mm_score near 1.0 ⇒ image-driven; near 0.0 ⇒ text-driven.
             Paper: Parcalabescu & Frank, ACL 2022 — https://arxiv.org/abs/2212.08158
             Code:  https://github.com/coastalcph/mm-shap

  VL-SHAP — Shapley attribution over a grid of spatial image regions.
             Ranks which regions most influenced the model's output logprob.
             Based on: Lundberg & Lee, NeurIPS 2017 — https://arxiv.org/abs/1705.07874
             Applied via MM-SHAP framework (Parcalabescu & Frank, ACL 2022)

Both require LOGPROBS capability; any OpenAI-compatible endpoint with
logprobs=True works (e.g. gpt-4o-mini with top_logprobs).

Usage (inside Docker):
    python run.py                     # uses config.yaml
    python run.py --model gpt-4o-mini

Expected output (values vary by model):
    [MM-SHAP] mm_score=0.62 (image-reliant), text_contribution=0.38, image_contribution=0.62
              top_text_tokens: [{"token": "color", "shapley": 0.12}, ...]
    [VL-SHAP] top_regions: [{"region": 4, "shapley": 0.31}, ...]  (region 4 = center)
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import yaml
from PIL import Image

from evalrx.analyzers.perturbation.mm_shap import MMShapAnalyzer
from evalrx.analyzers.perturbation.vl_shap import VLShapAnalyzer
from evalrx.core.case import CaseBatch, FailureCase as Case, Inputs

CONFIG = Path(__file__).parent / "config.yaml"


def _build_api_model(model_name: str):
    """API model with logprobs support (requires GEMINI_API_KEY)."""
    import openai

    from evalrx.models.backends.api import APIModel, parse_openai_logprobs
    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.core.spec import ModelSpec

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _img_b64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def generate_fn(prompt, *, model=model_name, image=None, **_):
        content = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_b64(image)}"}})
        content.append({"type": "text", "text": prompt})
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=64,
        )
        return resp.choices[0].message.content or ""

    def logprobs_fn(prompt, *, model=model_name, image=None, max_new_tokens=32, top_k=5, **_):
        content = []
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_b64(image)}"}})
        content.append({"type": "text", "text": prompt})
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens, logprobs=True, top_logprobs=top_k,
        )
        return parse_openai_logprobs(resp.choices[0].logprobs.content if resp.choices[0].logprobs else [])

    spec = ModelSpec(key=model_name, family="openai", model_type="api")
    rt = RuntimeConfig(generate_fn=generate_fn, logprobs_fn=logprobs_fn)
    return APIModel(spec, rt)


def _demo_image() -> Image.Image:
    """Gradient image — replace with a real photo for meaningful attributions."""
    img = Image.new("RGB", (128, 128))
    pixels = img.load()
    for y in range(128):
        for x in range(128):
            pixels[x, y] = (x * 2, y * 2, 128)
    return img


def run_mm_shap(model, cfg: dict) -> None:
    prompt = "What is the dominant color in this image?"
    case = Case(id="mm_shap_0", inputs=Inputs(prompt=prompt, image=_demo_image()))
    result = MMShapAnalyzer(
        n_samples=cfg.get("mm_shap_n_samples", 32),
        top_k=cfg.get("mm_shap_top_k", 5),
    ).run(model, CaseBatch([case]))
    print("[MM-SHAP]", result.summary())
    f = result.findings
    print(f"  mm_score={f['mm_score']} (0=text-reliant, 1=image-reliant)")
    print(f"  text_contribution={f['text_contribution']}, image_contribution={f['image_contribution']}")
    print(f"  top_text_tokens: {f['top_text_tokens']}")


def run_vl_shap(model, cfg: dict) -> None:
    prompt = "Describe what you see."
    case = Case(id="vl_shap_0", inputs=Inputs(prompt=prompt, image=_demo_image()))
    result = VLShapAnalyzer(
        n_regions=cfg.get("vl_shap_n_regions", 9),
        n_samples=cfg.get("vl_shap_n_samples", 32),
        top_k=cfg.get("vl_shap_top_k", 3),
    ).run(model, CaseBatch([case]))
    print("[VL-SHAP]", result.summary())
    f = result.findings
    print(f"  grid: {f['grid_side']}x{f['grid_side']}, total_abs_attribution={f['total_abs_attribution']}")
    print(f"  top_regions: {f['top_regions']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = args.model or cfg.get("model_name", "gpt-4o-mini")

    model = _build_api_model(model_name)
    run_mm_shap(model, cfg)
    run_vl_shap(model, cfg)


if __name__ == "__main__":
    main()
