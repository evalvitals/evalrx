"""L3b fix executors — internals-**modifying** repairs (the sandbox boundary).

This module holds only what genuinely cannot be handed to a sandboxed coding
agent: pre-audited, parameterised primitives that **write** to the forward pass.

* **L3a (internals, read)** is intentionally NOT here at all.  Reading attention
  needs no privileged model handle, so the capability is exposed to sandboxed
  coded pipelines via ``model_attend()`` (see :mod:`fix_pipeline`), and the
  agent writes its own peak-find → crop → re-ask scaffold.  The host-side
  capture that backs ``model_attend()`` is
  :func:`~evalrx.analyzers.attention.relative_attn.attention_heatmap` — a
  generic attention reducer that lives with the analyzers (one reduction shared
  with the white-box probe path), not in this fix module.  (An earlier
  ``attention_guided_crop`` primitive was removed for the same reason — it
  duplicated what the coded path already writes.)
* **L3b (internals, write)**: pre-audited intervention primitives that modify the
  forward pass — **never** free codegen against the model handle, because
  arbitrary hook code with the raw model object cannot be sandboxed.  The judge
  selects and parameterises; it never authors these.  v1 ships *visual embedding
  boost*: a forward hook on the input-embedding layer scaling image-token
  embeddings by ``gamma`` (architecture-agnostic for HF VLMs whose image tokens
  are placeholder ids in ``input_ids``).  Attention-map editing needs
  per-architecture hooks and joins this registry later.

L4 (parameter space): :class:`FinetuneSpec` captures a complete fine-tune
recipe (dataset construction generalising the verified hypothesis, method,
target, evaluation protocol incl. a regression battery). v1 of the executor
(:func:`run_lora_repair`) executes exactly one recipe shape — ``method="lora"``
on ``target="llm"`` — and refuses everything else with a descriptive reason
rather than guessing:

* ``dataset_recipe`` (free natural-language text) is **never interpreted**.
  There is no safe path from an arbitrary judge-written string to executable
  data synthesis without another judge call producing code — the same
  structured-output path whose failures cost three real bugs elsewhere in
  this project (see the ``vlm_paper_benchmark`` example's README). Training
  data is instead a fixed, defensible default: diagnosis-split failing cases
  as ``(prompt, image) -> gold answer`` SFT pairs, plus a sample of
  diagnosis-split *passing* cases as anti-forgetting ballast — supplied by
  the caller via a pool that must be disjoint from whatever batch is being
  validated (see :class:`~.fix_agent.FixAgent`'s ``finetune_pool``).
* ``method in {"sft", "full"}`` and ``target in {"vision_encoder",
  "projector"}`` are recorded (for the escalation decision) but not
  executed — the former needs a real SFT training loop rather than a
  handful of LoRA steps, the latter needs per-architecture module-naming
  conventions this project has no way to validate without a training run
  per architecture. Both are TODO.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalrx.eval_agent.stages.fix_tiers import FixTier
from evalrx.eval_agent.stages.fix_tools import score_to_bool

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch, FailureCase
    from evalrx.core.model import Model

logger = logging.getLogger(__name__)

ScoreFn = Callable[["FailureCase", str], Optional[bool]]


# ---------------------------------------------------------------------------
# L3b — visual embedding boost (forward-hook intervention)
# ---------------------------------------------------------------------------

def _resolve_hf(model: "Model"):
    """Underlying (HF model, image_token_id) or (None, None) when unavailable."""
    hf = getattr(model, "_hf", None)
    hf_model = hf[0] if isinstance(hf, tuple) and hf else None
    if hf_model is None:
        return None, None
    cfg = getattr(hf_model, "config", None)
    for attr in ("image_token_id", "image_token_index"):
        tid = getattr(cfg, attr, None)
        if tid is not None:
            return hf_model, int(tid)
    return None, None


def boost_available(model: "Model") -> bool:
    return _resolve_hf(model)[0] is not None


@contextmanager
def visual_embedding_boost(model: "Model", gamma: float = 1.5):
    """Scale image-token embeddings by *gamma* for every forward inside the block."""
    hf_model, image_token_id = _resolve_hf(model)
    if hf_model is None:
        raise RuntimeError(
            "visual_embedding_boost: backend internals unavailable "
            "(needs a loaded hf_local model exposing image_token_id)"
        )
    embedding = hf_model.get_input_embeddings()
    gamma = float(gamma)

    def _hook(module, args, output):
        input_ids = args[0]
        mask = input_ids == image_token_id
        if mask.any():
            output = output.clone()
            output[mask] = output[mask] * gamma
        return output

    handle = embedding.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


def run_visual_embedding_boost(
    model: "Model",
    cases: "CaseBatch",
    score_fn: ScoreFn,
    params: "dict[str, Any] | None" = None,
) -> "dict[str, Optional[bool]]":
    """Generate every case under the boost hook; score against the rubric."""
    gamma = float((params or {}).get("gamma", 1.5))
    scores: "dict[str, Optional[bool]]" = {}
    try:
        with visual_embedding_boost(model, gamma=gamma):
            for case in cases:
                try:
                    out = str(model.generate(case.inputs))
                except Exception as exc:
                    logger.debug("boosted generate failed on %s: %s", case.id, exc)
                    scores[case.id] = None
                    continue
                scores[case.id] = score_to_bool(score_fn(case, out))
    except RuntimeError as exc:
        logger.warning("visual_embedding_boost unavailable: %s", exc)
        return {c.id: None for c in cases}
    return scores


# ---------------------------------------------------------------------------
# Primitive registry (judge selects + parameterises; code is pre-audited)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InternalsPrimitive:
    """A host-side, pre-audited L3 intervention primitive."""

    name: str
    tier: FixTier
    description: str
    params_hint: str
    available: "Callable[[Model], bool]"
    run: "Callable[[Model, CaseBatch, ScoreFn, dict | None], dict[str, Optional[bool]]]"


#: Pre-audited internals-WRITE primitives only.  Reads (L3a) are not here — the
#: agent authors them against the ``model_attend()`` bridge (see module docstring).
INTERNALS_PRIMITIVES: "dict[str, InternalsPrimitive]" = {
    "visual_embedding_boost": InternalsPrimitive(
        name="visual_embedding_boost",
        tier=FixTier.L3B_INTERNALS_WRITE,
        description="scale image-token embeddings by gamma via a forward hook "
                    "(amplifies visual evidence against language priors)",
        params_hint='{"gamma": float > 1 (default 1.5)}',
        available=boost_available,
        run=run_visual_embedding_boost,
    ),
}


def primitives_catalog_text(model: "Model", max_tier: FixTier) -> str:
    """Render available primitives (≤ *max_tier*, supported by *model*)."""
    lines = []
    for prim in INTERNALS_PRIMITIVES.values():
        if prim.tier <= max_tier and prim.available(model):
            lines.append(f"- {prim.name} [{prim.tier.label}]: {prim.description}  "
                         f"[params: {prim.params_hint}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# L4 — parameter space: recipe dataclass + v1 LoRA executor
# ---------------------------------------------------------------------------

@dataclass
class FinetuneSpec:
    """A complete L4 fine-tune recipe.

    Executed by :func:`run_lora_repair` when ``method == "lora"`` and
    ``target == "llm"`` and the caller supplied a ``finetune_pool``; recorded
    without running otherwise (TODO: SFT training loop for ``method="sft"``/
    ``"full"``; per-architecture module-naming for ``target="vision_encoder"``/
    ``"projector"``). :class:`~.fix_agent.FixAgent` always records the recipe
    as a candidate either way, so the escalation decision has something
    concrete to act on even when it wasn't executed.

    Attributes:
        dataset_recipe: How to build training data that *generalises* the
                        verified mechanism (never just the failing cases).
        method:         Training method, e.g. ``"lora"`` / ``"sft"``.
        target:         Component to tune, e.g. ``"vision_encoder"`` /
                        ``"llm"`` / ``"projector"`` / ``"full"``.
        eval_protocol:  How the tuned model must be validated — held-out
                        repair effect AND no-regression on passing cases.
        rationale:      Why parameter-space change is the minimum effective
                        intervention for the hypothesis.
    """

    dataset_recipe: str
    method: str = "lora"
    target: str = "llm"
    eval_protocol: str = (
        "paired McNemar on held-out failures + regression battery on all "
        "baseline-passing cases"
    )
    rationale: str = ""
    metadata: "dict[str, Any]" = field(default_factory=dict)

    def to_dict(self) -> "dict[str, Any]":
        return {
            "dataset_recipe": self.dataset_recipe,
            "method": self.method,
            "target": self.target,
            "eval_protocol": self.eval_protocol,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class LoraRepairResult:
    """Outcome of one L4 LoRA train-and-validate cycle.

    Mirrors :class:`~.fix_pipeline.CodedPipelineResult`'s shape (``scores``/
    ``ok``/``error``) so the caller's ``exec_error`` accounting is identical
    across coded and fine-tune candidates.
    """

    scores: "dict[str, Optional[bool]]" = field(default_factory=dict)
    ok: bool = False
    error: str = ""


def lora_available() -> bool:
    """Whether the optional ``peft`` dependency (``pip install evalrx[finetune]``)
    is installed. ``peft`` hard-depends on ``transformers``, so this single
    check also covers that."""
    try:
        import peft  # noqa: F401
    except ImportError:
        return False
    return True


def _lora_target_modules(hf_model: Any) -> "str | list[str]":
    """Scope LoRA to the language-model component (v1 only ever executes
    ``target="llm"``).

    When the model exposes a ``language_model`` submodule (the same
    convention :func:`~.hf_local.HFLocalModel.generate_ifcd` and
    ``generate_pai`` already use for "the LLM half of a VLM"), a regex
    anchored to that dotted prefix is used — plain name-suffix matching
    (``q_proj``/``k_proj``/...) is not safe here, because CLIP-style vision
    towers commonly use the *same* projection names, and a suffix-only match
    would silently pull the vision encoder into a candidate whose contract
    promises ``target="llm"`` only. Falls back to plain suffix matching for a
    text-only backbone, where there is no vision tower to leak into.
    """
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    if getattr(hf_model, "language_model", None) is not None:
        return r"language_model\..*\.(" + "|".join(suffixes) + ")"
    return list(suffixes)


def _build_sft_examples(
    train_cases: "CaseBatch", max_examples: int
) -> "list[tuple[FailureCase, str]]":
    """Diagnosis-split-only SFT pairs — never interprets ``dataset_recipe``
    (see module docstring). Failing cases (gold answer as the SFT target)
    teach the correction; a matched sample of already-passing cases is
    anti-forgetting ballast. The caller (:class:`~.fix_agent.FixAgent`) is
    responsible for ``train_cases`` being disjoint from whatever batch is
    being validated — this function has no way to detect contamination.
    """
    from evalrx.core.case import Label

    def _usable(case: "FailureCase") -> bool:
        return case.expected not in (None, "") and bool(str(case.inputs.prompt or "").strip())

    fail = [(c, str(c.expected)) for c in train_cases if c.label == Label.FAIL and _usable(c)]
    passed = [(c, str(c.expected)) for c in train_cases if c.label == Label.PASS and _usable(c)]
    if not fail:
        return []
    budget = max(1, int(max_examples))
    n_fail = min(len(fail), budget)
    n_pass = min(len(passed), max(0, budget - n_fail))
    return fail[:n_fail] + passed[:n_pass]


def _encode_sft_example(
    processor: Any, prompt: str, image: Any, target_text: str, chat_template_kwargs: "dict[str, Any]"
) -> "dict[str, Any]":
    """One teacher-forced training example: everything up to (and including)
    the prompt is masked to ``-100``; loss applies only to the target-answer
    tokens.

    Uses the processor's chat template when available (production VLM/LLM
    processors); falls back to plain ``prompt + " " + target`` concatenation
    otherwise (simple tokenizers, InstructBLIP-style processors with no chat
    template — its ``Question:``/``Answer:`` framing is not reproduced here,
    a v1 simplification). The prompt/target boundary is found by tokenizing
    the prompt-only prefix separately and using its length as the mask
    cutoff: an approximation (BPE merges spanning the boundary can shift a
    token or two) accepted as good enough for v1.
    """
    tok = getattr(processor, "tokenizer", processor)
    chat_template = getattr(tok, "chat_template", None)
    images = None
    if image is not None:
        images = list(image) if isinstance(image, (list, tuple)) else [image]

    if chat_template:
        content: Any = (
            [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
            if images else prompt
        )
        prefix_text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **chat_template_kwargs,
        )
        full_text = processor.apply_chat_template(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": target_text},
            ],
            add_generation_prompt=False, tokenize=False, **chat_template_kwargs,
        )
        if images:
            prefix_enc = processor(text=[prefix_text], images=images, return_tensors="pt")
            full_enc = processor(text=[full_text], images=images, return_tensors="pt")
        else:
            prefix_enc = tok(prefix_text, return_tensors="pt")
            full_enc = tok(full_text, return_tensors="pt")
    else:
        prefix_enc = tok(prompt, return_tensors="pt")
        full_enc = tok(prompt + " " + target_text, return_tensors="pt")

    prefix_len = prefix_enc["input_ids"].shape[1]
    labels = full_enc["input_ids"].clone()
    labels[:, :prefix_len] = -100
    enc = dict(full_enc)
    enc["labels"] = labels
    enc.pop("token_type_ids", None)  # some VLM processors emit this; generate()/forward() reject it
    return enc


def run_lora_repair(
    model: "Model",
    train_cases: "CaseBatch | None",
    val_cases: "CaseBatch",
    spec_payload: "dict[str, Any]",
    score_fn: ScoreFn,
    *,
    max_train_examples: int = 16,
    max_steps: int = 30,
    lr: float = 1e-3,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> LoraRepairResult:
    """Train a LoRA adapter on ``train_cases`` (diagnosis-split only — MUST be
    disjoint from ``val_cases``, the batch being validated) and generate on
    ``val_cases`` with it applied, so the caller's existing McNemar + e-value
    machinery validates L4 exactly like every other tier — the
    ``eval_protocol`` field's "held-out repair effect" half is satisfied by
    that comparison; the "regression battery on baseline-passing cases" half
    is ``n_broken`` on whatever baseline-passing cases the caller's ``data``
    batch includes, which :meth:`~.fix_agent.FixAgent._validate` already
    computes. No second validation path is built here.

    Executes exactly one recipe shape (``method="lora"``, ``target="llm"``);
    everything else returns ``ok=False`` with a descriptive ``error`` (see
    module docstring). LoRA is spliced into ``model``'s underlying HF module
    *in place* (via ``peft.get_peft_model``) and unloaded again in a
    ``finally`` — the model is back to its exact pre-call state (verified by
    this module's tests) whether or not this call ever reaches generation.
    """
    method = str(spec_payload.get("method", "lora")).strip().lower()
    target = str(spec_payload.get("target", "llm")).strip().lower()
    if method != "lora":
        return LoraRepairResult(
            error=f"only method='lora' is executed; method={method!r} is TODO "
            "(needs a real SFT training loop, not a handful of LoRA steps)"
        )
    if target != "llm":
        return LoraRepairResult(
            error=f"only target='llm' is executed; target={target!r} is TODO "
            "(needs per-architecture module-naming this project has no way to "
            "validate without a training run per architecture)"
        )
    if not lora_available():
        return LoraRepairResult(error="peft is not installed (pip install evalrx[finetune])")
    if train_cases is None or len(train_cases) == 0:
        return LoraRepairResult(
            error="no finetune_pool configured — FixAgent(finetune_pool=...) needs a "
            "diagnosis-only CaseBatch, never the validation split"
        )
    # NOT _resolve_hf(): that helper also requires an image_token_id (it backs
    # the vision-only visual_embedding_boost primitive) and would reject any
    # text-only backbone -- exactly the fallback target_modules case below.
    hf = getattr(model, "_hf", None)
    hf_model = hf[0] if isinstance(hf, tuple) and len(hf) == 2 else None
    processor = hf[1] if isinstance(hf, tuple) and len(hf) == 2 else None
    if hf_model is None or processor is None:
        return LoraRepairResult(
            error="backend internals unavailable (needs a loaded hf_local model + processor)"
        )

    examples = _build_sft_examples(train_cases, max_train_examples)
    if not examples:
        return LoraRepairResult(
            error="finetune_pool has no failing case with a known gold answer to train on"
        )

    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    chat_template_kwargs = dict(getattr(getattr(model, "spec", None), "chat_template_kwargs", {}) or {})
    target_modules = _lora_target_modules(hf_model)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM,
    )
    was_training = hf_model.training
    device = next(hf_model.parameters()).device
    # get_peft_model() itself raises (before injecting anything, so hf_model
    # is untouched) when target_modules matches zero layers -- e.g. a VLM
    # whose language_model uses non-LLaMA-style projection names. That must
    # not propagate as an uncaught exception out of one candidate's
    # validation and abort the whole FixAgent run.
    try:
        peft_model = get_peft_model(hf_model, lora_cfg)
    except Exception as exc:
        return LoraRepairResult(
            error=f"no matching linear layers for target_modules={target_modules!r} "
            f"on this architecture: {exc}"
        )
    try:
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        if not trainable:  # defensive belt-and-braces; get_peft_model already guards this
            return LoraRepairResult(
                error=f"no matching linear layers for target_modules={target_modules!r} "
                "on this architecture"
            )
        peft_model.train()
        opt = torch.optim.AdamW(trainable, lr=lr)
        n_steps_run = 0
        for step in range(max_steps):
            case, target_text = examples[step % len(examples)]
            try:
                enc = _encode_sft_example(
                    processor, str(case.inputs.prompt or ""), getattr(case.inputs, "image", None),
                    target_text, chat_template_kwargs,
                )
            except Exception as exc:
                logger.debug("run_lora_repair: encode failed on %s: %s", case.id, exc)
                continue
            enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
            opt.zero_grad()
            try:
                out = peft_model(**enc)
                if out.loss is None:
                    continue
                out.loss.backward()
                opt.step()
                n_steps_run += 1
            except Exception as exc:
                logger.debug("run_lora_repair: training step failed on %s: %s", case.id, exc)
                continue
        if n_steps_run == 0:
            return LoraRepairResult(error="every training step failed to encode/run (see debug log)")

        peft_model.eval()
        scores: "dict[str, Optional[bool]]" = {}
        with torch.no_grad():
            for case in val_cases:
                try:
                    out = str(model.generate(case.inputs))
                except Exception as exc:
                    logger.debug("run_lora_repair: generate failed on %s: %s", case.id, exc)
                    scores[case.id] = None
                    continue
                scores[case.id] = score_to_bool(score_fn(case, out))
        return LoraRepairResult(scores=scores, ok=True)
    finally:
        peft_model.unload()
        hf_model.train(was_training)
