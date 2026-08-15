"""Stage W — model internals for the handful of cases M3 actually points at.

The endpoint that generates the batch returns text, not internals, so every
ATTENTION-requiring analyzer is silently skipped in the main chain (see the
capability table in README.md). This module re-forwards a SMALL selected subset
through transformers so those analyzers have something to read.

    python whitebox.py --model qwen3.5-9b --dataset supergpqa_law --probe
    python run_whitebox.py --model qwen3.5-9b --dataset supergpqa_law --n 24

Two hard constraints shape it, and both are the reason this is a separate stage
rather than a flag on the main one:

1. **A different interpreter.** ``qwen3_5`` is unknown to transformers 4.57.6
   (the version in the evalvitals venv) and known to 5.15.0 (the version in the
   vLLM venv). Stage W therefore runs under ``WHITEBOX_PYTHON``, exactly as
   serving already runs under ``VLLM_BIN``. :func:`require_transformers` checks
   this at import rather than letting ``from_pretrained`` fail 20 GB later.

2. **Attention is O(layers x heads x seq^2).** For the 9B (32 layers, 16 heads)
   a 2k-token prompt costs ~8.6 GB per case, a 4k-token prompt ~34 GB. Analyzers
   call ``forward(capture={ATTENTION})`` with no CaptureSpec — i.e. they ask for
   everything — so :class:`BoundedWhitebox` intercepts that and REFUSES past a
   budget instead of OOM-ing. It does not quietly subset layers: attention
   rollout multiplies through the full stack, so a layer subset changes what
   rollout means rather than just how much it costs.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

#: Bytes of attention a single forward may materialise before we refuse.
#: 8 GB ~ a 2k-token prompt on the 9B. Raise it only with the RAM to back it:
#: ``to_cpu=True`` means this lands in host memory, not on the card.
DEFAULT_ATTN_BUDGET_BYTES = 8 * 1024 ** 3

#: bf16 attention weights are 2 bytes; assume 4 so the guard errs toward refusing.
_BYTES_PER_WEIGHT = 4


def require_transformers() -> str:
    """Fail loudly, now, if this interpreter cannot load a Qwen3.5 checkpoint.

    ``ModelSpec.min_transformers`` is declared but nothing in the framework
    enforces it, and the failure it prevents is expensive and confusing: an old
    transformers raises a bare ``KeyError: 'qwen3_5'`` from deep inside
    ``AutoConfig``, which reads like a corrupt download rather than a wrong venv.
    """
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if "qwen3_5" not in CONFIG_MAPPING_NAMES:
        raise SystemExit(
            f"transformers {transformers.__version__} does not know 'qwen3_5'.\n"
            f"  Stage W needs the interpreter that serves the model, not the one "
            f"that runs the pipeline:\n"
            f"    WHITEBOX_PYTHON=/path/to/vllm-venv/bin/python ./run_all.sh ...\n"
            f"  (verified working: transformers 5.15.0; absent in 4.57.6)"
        )
    return transformers.__version__


def attention_bytes(seq_len: int, n_layers: int, n_heads: int) -> int:
    return n_layers * n_heads * seq_len * seq_len * _BYTES_PER_WEIGHT


class BoundedWhitebox:
    """An hf_local model whose ``forward`` refuses to materialise the impossible.

    Everything except ``forward`` is delegated untouched, so it is a drop-in for
    any analyzer. ``forward`` estimates the attention tensor first and raises a
    :class:`MemoryError` naming the sequence length and the budget — a message
    an operator can act on, unlike the CUDA/host OOM that would otherwise land.
    """

    def __init__(self, inner, budget_bytes: int = DEFAULT_ATTN_BUDGET_BYTES,
                 layers: "list[int] | None" = None, to_cpu: bool = True):
        self._inner = inner
        self.budget_bytes = budget_bytes
        self.layers = layers
        self.to_cpu = to_cpu
        self.n_forwards = 0
        self.max_seq_seen = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"BoundedWhitebox({self._inner!r})"

    def attention_layers(self) -> list:
        """Model-layer indices that actually produce an attention matrix.

        Qwen3.5 is a HYBRID stack: ``layer_types`` reads
        ``[linear_attention, linear_attention, linear_attention, full_attention]``
        repeated, so 32 layers yield 8 tensors. Those 8 are dense [H, Q, K]
        matrices and perfectly readable — but the list is not indexed by layer,
        and a finding that says "layer 2" without this mapping names a layer that
        has no attention at all.
        """
        hf, _ = self._inner._loaded
        conf = getattr(hf, "config", None)
        text = getattr(conf, "text_config", None) or conf
        types = list(getattr(text, "layer_types", []) or [])
        if types:
            return [i for i, t in enumerate(types) if "full" in str(t)]
        return list(range(int(getattr(text, "num_hidden_layers", 0))))

    def _shape(self) -> tuple:
        """(n_capturable_layers, n_heads) — capturable, not total depth."""
        hf, _ = self._inner._loaded
        conf = getattr(hf, "config", None)
        text = getattr(conf, "text_config", None) or conf
        return len(self.attention_layers()), int(getattr(text, "num_attention_heads", 0))

    def forward(self, inputs, capture, spec=None):
        from evalvitals.core.capability import Capability
        from evalvitals.core.model import CaptureSpec

        if Capability.ATTENTION in set(capture):
            n_layers, n_heads = self._shape()
            kept = len(self.layers) if self.layers is not None else n_layers
            seq = self._token_count(inputs)
            self.max_seq_seen = max(self.max_seq_seen, seq)
            need = attention_bytes(seq, kept, n_heads)
            if need > self.budget_bytes:
                raise MemoryError(
                    f"attention for a {seq}-token prompt over {kept} layers x "
                    f"{n_heads} heads needs ~{need / 1024 ** 3:.1f} GB, budget is "
                    f"{self.budget_bytes / 1024 ** 3:.1f} GB.\n"
                    f"  Options, in order of how little they cost you:\n"
                    f"   - raise --attn-budget-gb if the host has the RAM;\n"
                    f"   - pick shorter cases (--max-prompt-tokens drops the rest);\n"
                    f"   - --layers 0,8,16,24,31 to capture a subset — but note "
                    f"attention_rollout composes through the WHOLE stack, so its "
                    f"output stops meaning what its name says."
                )
        if spec is None:
            spec = CaptureSpec(layers=self.layers, to_cpu=self.to_cpu)
        self.n_forwards += 1
        return self._inner.forward(inputs, capture, spec)

    def _token_count(self, inputs) -> int:
        _, processor = self._inner._loaded
        tok = getattr(processor, "tokenizer", processor)
        text = str(getattr(inputs, "prompt", inputs))
        try:
            return len(tok(text)["input_ids"])
        except Exception:
            return len(text) // 3  # crude, only feeds the guard


def build(model_id: str, *, device: str = "cuda", dtype: str = "bfloat16",
          budget_bytes: int = DEFAULT_ATTN_BUDGET_BYTES,
          layers: "list[int] | None" = None) -> BoundedWhitebox:
    """Compose the hf_local white-box model for *model_id* (a spec key).

    ``device="cuda"`` rather than ``"auto"`` on purpose: the auto path goes
    through ``device_map``, which requires ``accelerate`` — absent from the vLLM
    venv Stage W runs in. One card is the right assumption anyway, since the
    whole pipeline already serves on one. Pass ``device="auto"`` explicitly if
    you have accelerate and a model that needs sharding.
    """
    from evalvitals.core.capability import Capability
    from evalvitals.models.backends.base import RuntimeConfig
    from evalvitals.models.compose import compose

    require_transformers()
    model = compose(
        model_id, "hf_local",
        runtime=RuntimeConfig(device=device, dtype=dtype, attn_impl="eager"),
        want={Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES},
    )
    return BoundedWhitebox(model, budget_bytes=budget_bytes, layers=layers)


def select_cases(report: dict, n: int, seed: int = 0,
                 max_prompt_tokens: int | None = None,
                 tokenizer=None) -> list:
    """A label-balanced subset of the frozen batch.

    Balanced on purpose: every attention analyzer here contrasts FAIL against
    PASS, and a subset drawn by accuracy would hand a 0.70-accuracy slice only
    a third as many FAILs as PASSes — the comparison M3 asked for, run at a
    fraction of the power, without saying so.
    """
    cases = report["cases"]
    if max_prompt_tokens and tokenizer is not None:
        cases = [c for c in cases
                 if len(tokenizer(c["prompt"])["input_ids"]) <= max_prompt_tokens]
    rng = random.Random(seed)
    out = []
    for label in ("FAIL", "PASS"):
        pool = [c for c in cases if c["label"] == label]
        rng.shuffle(pool)
        out.extend(pool[: n // 2])
    rng.shuffle(out)
    return out


def to_batch(cases: list):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    return CaseBatch([
        FailureCase(
            inputs=Inputs(prompt=c["prompt"]),
            observed=c["output"],
            expected=c["gold"] if not isinstance(c["gold"], list) else c["gold"][0],
            label=Label.PASS if c["label"] == "PASS" else Label.FAIL,
        )
        for c in cases
    ])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="probe the Stage W environment")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--probe", action="store_true", help="check env, do not load weights")
    args = ap.parse_args()

    version = require_transformers()
    print(f"transformers {version}: qwen3_5 supported")
    from evalvitals.specs import get_spec

    spec = get_spec(args.model)
    print(f"spec {spec.key}: {spec.hf_repo} via {spec.auto_class}")
    for c in spec.caveats:
        print(f"  caveat: {c}")
    if not args.probe:
        model = build(args.model)
        print(f"loaded {model!r} caps={sorted(c.value for c in model.capabilities)}")
        n_layers, n_heads = model._shape()
        print(f"  {n_layers} layers x {n_heads} heads")
        for seq in (512, 1024, 2048, 4096):
            gb = attention_bytes(seq, n_layers, n_heads) / 1024 ** 3
            fits = "ok" if gb * 1024 ** 3 <= model.budget_bytes else "REFUSED"
            print(f"  seq {seq:5d} -> {gb:6.2f} GB  {fits}")
