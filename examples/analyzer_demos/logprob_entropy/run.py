"""Logprob entropy analysis example — black-box perplexity + uncertainty.

Demonstrates LogprobEntropyAnalyzer: uses output-token logprobs (OpenAI-style)
to compute sequence perplexity and per-token predictive entropy without needing
white-box access to model internals.

  perplexity        = exp(-mean_logprob) — lower is more confident
  mean_top_entropy  = mean Shannon entropy over top-k token distribution per step

A high perplexity or entropy on early tokens typically signals uncertainty.
Compare across prompts or models to detect knowledge gaps or ambiguous inputs.

References:
  Predictive entropy: Gal & Ghahramani, ICML 2016 — https://arxiv.org/abs/1506.02142
  LLM self-knowledge: Kadavath et al. 2022    — https://arxiv.org/abs/2207.05221

Usage (inside Docker):
    python run.py                     # uses config.yaml
    python run.py --prompt "..."

Expected output (values vary):
    [logprob_entropy] n_tokens=12, perplexity=4.21, mean_logprob=-1.44,
                      min_token_logprob=-3.82, mean_top_entropy=0.73
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from evalrx.analyzers.uncertainty.logprob_entropy import LogprobEntropyAnalyzer
from evalrx.core.case import CaseBatch, FailureCase as Case, Inputs

CONFIG = Path(__file__).parent / "config.yaml"

DEMO_PROMPTS = [
    "The capital of France is",
    "The speed of light in a vacuum is approximately",
    "Who won the 2024 US presidential election?",
]


def _build_api_model(model_name: str, max_new_tokens: int, top_k: int):
    import openai

    from evalrx.core.spec import ModelSpec
    from evalrx.models.backends.api import APIModel, parse_openai_logprobs
    from evalrx.models.backends.base import RuntimeConfig

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_fn(prompt, *, model=model_name, **_):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content or ""

    def logprobs_fn(prompt, *, model=model_name, **_):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            logprobs=True,
            top_logprobs=top_k,
        )
        lp = resp.choices[0].logprobs
        return parse_openai_logprobs(lp.content if lp else [])

    spec = ModelSpec(key=model_name, family="openai", model_type="api")
    rt = RuntimeConfig(generate_fn=generate_fn, logprobs_fn=logprobs_fn)
    return APIModel(spec, rt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = _build_api_model(
        cfg.get("model_name", "gpt-4o-mini"),
        cfg.get("max_new_tokens", 40),
        cfg.get("top_k", 5),
    )

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS
    analyzer = LogprobEntropyAnalyzer()

    for prompt in prompts:
        case = Case(id="lpe", inputs=Inputs(prompt=prompt))
        result = analyzer.run(model, CaseBatch([case]))
        f = result.findings
        print(f"Prompt: {prompt!r}")
        print(f"  perplexity={f['perplexity']}, mean_logprob={f['mean_logprob']}, "
              f"mean_top_entropy={f['mean_top_entropy']}, n_tokens={f['n_tokens']}")
        print()


if __name__ == "__main__":
    main()
