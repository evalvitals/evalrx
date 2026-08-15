#!/usr/bin/env python3
"""White-box VLM paper runner for decoding/intervention paper methods.

This intentionally shares the frozen-data protocol and scorer with
``run_autofix.py`` while loading an ``HFLocalModel`` directly.  It is the
execution route for methods such as VCD and ICD that require model logits
rather than an OpenAI-compatible endpoint approximation.

``FixAgent`` defaults to ``max_repair_rounds=1``, which this runner never
overrides.  That is not a truncated loop: one round already proposes
candidates across every tier up to ``--max-tier`` in a single pass, so an
unpinned run explores the full L1/L2/L3 ladder in round 1.  A
``--only-paper-candidate`` run pinning one name will always show
``repair_rounds: 1`` in its report for a different reason -- round 2 would
re-propose that same paper-default candidate, the dedup set drops it as
already-seen, and the loop stops with no new candidate.  Neither case is a
wiring bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_autofix import (
    OUT,
    PAPER_IDS,
    PAPER_SPECS,
    diagnostic_split,
    evaluate,
    exclude_seen_rows,
    load_records,
    load_records_from,
    make_cases,
    score_case,
    shuffled_rows,
    task_prompt,
)

from evalvitals.core.case import Inputs
from evalvitals.eval_agent.hypothesis import Hypothesis, hypothesis_to_dict
from evalvitals.eval_agent.stages.fix_agent import FixAgent
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.specs import get_spec

# These names are deliberately the executor's stable candidate identifiers,
# not paper titles.  Validate them before an expensive model run so a typo or
# a stale README command cannot turn a paper reproduction into an empty trial.
PAPER_CANDIDATE_NAMES = frozenset(
    {
        "vcd_diffusion_noise",
        "vcd_diffusion_noise_gated_false_yes",
        "icd_instruction_disturbance",
        "icd_instruction_disturbance_question",
        "icd_instruction_disturbance_gated_false_yes",
        "opera_overtrust_binary",
        "ifcd_truthx_contrast",
        "vicrop_relative_attention",
        "vicrop_consensus_guard",
        "pai_image_attention",
    }
)


def artifact_sha256(path: str | None) -> str | None:
    """Fingerprint a local method artifact without copying it into a report."""
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_pope_prompt(row: dict[str, object]) -> str:
    """Prompt contract of VCD's released LLaVA POPE evaluator.

    The release appends this one-word instruction before rendering its LLaVA
    conversation template.  It is intentionally distinct from EvalVitals'
    generic strict Yes/No prompt so a paper-method report records the actual
    task contract it tested.
    """
    return str(row["question"]) + " Please answer this question with one word."


_CANDIDATE_PROMPT_CONTRACTS: dict[str, tuple[str, object]] = {
    "vcd_diffusion_noise": ("vcd_pope_question_plus_one_word", source_pope_prompt),
    "vcd_diffusion_noise_gated_false_yes": (
        "vcd_pope_question_plus_one_word",
        source_pope_prompt,
    ),
    "icd_instruction_disturbance": ("pope_raw_question", lambda row: str(row["question"])),
    "icd_instruction_disturbance_question": (
        "pope_raw_question",
        lambda row: str(row["question"]),
    ),
    "icd_instruction_disturbance_gated_false_yes": (
        "pope_raw_question",
        lambda row: str(row["question"]),
    ),
    "opera_overtrust_binary": ("pope_raw_question", lambda row: str(row["question"])),
    "ifcd_truthx_contrast": ("pope_raw_question", lambda row: str(row["question"])),
    "pai_image_attention": ("pope_raw_question", lambda row: str(row["question"])),
}


def paper_prompt_contract(candidate_names: list[str]) -> tuple[str, object]:
    """Return the evaluator prompt contract shared by the pinned candidates.

    A gated candidate (e.g. ``vcd_diffusion_noise_gated_false_yes``) uses the
    exact same released prompt contract as its ungated sibling -- only its
    per-case applicability differs -- so pinning both together to compare
    them under one selection run is legitimate; pinning candidates whose
    *prompt contracts* differ (e.g. VCD with ICD) is not, since that would
    silently average two different task setups into one report.
    """
    contracts = {}
    for name in candidate_names:
        if name not in _CANDIDATE_PROMPT_CONTRACTS:
            raise SystemExit(f"--paper-prompt has no source prompt contract for {name}")
        contract_name, prompt_fn = _CANDIDATE_PROMPT_CONTRACTS[name]
        contracts[contract_name] = prompt_fn
    if len(contracts) != 1:
        raise SystemExit(
            "--paper-prompt requires every --only-paper-candidate to share one "
            "released prompt contract; got: " + ", ".join(sorted(contracts))
        )
    (contract_name, prompt_fn), = contracts.items()
    return contract_name, prompt_fn


def diagnose_hf(
    model: HFLocalModel, baseline_probe: dict[str, object], diagnosis_rows: list[dict[str, object]]
) -> str | None:
    """Ask the model itself to infer a failure mechanism from real failures.

    Mirrors ``run_autofix.py``'s ``diagnose()``: without this, the runner
    only ever fed FixAgent a hand-written one-line hypothesis, and no
    judge was wired in either (see ``_JudgeModel`` below) -- so every prior
    run proposed the same fixed default candidates regardless of what the
    diagnosis split actually showed. Returns ``None`` (caller keeps its
    prior text) when there is nothing to diagnose or the call errors out;
    this must never raise and abort an expensive validated run.
    """
    cases_by_id = {case["id"]: case for case in baseline_probe["cases"]}
    failures = [
        (row, cases_by_id[row["id"]])
        for row in diagnosis_rows
        if row["id"] in cases_by_id and not cases_by_id[row["id"]]["correct"]
    ]
    if not failures:
        return None
    failure_rows = [
        {"question": str(row["question"])[:280], "output": str(case["output"])[:280]}
        for row, case in failures[:8]
    ]
    representative, _ = failures[0]
    try:
        text = model.generate(
            Inputs(
                "Infer one narrow visual failure mechanism from these incorrect "
                "image-question responses. The attached image is the first "
                "failure: inspect it to ground the diagnosis. Do not propose a "
                "fix or mention unavailable hidden model state.\n\n"
                + json.dumps(failure_rows, ensure_ascii=False),
                representative["image"],
            ),
            max_new_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 - diagnosis is best-effort, never fatal
        print(f"diagnose_hf: model call failed, keeping prior hypothesis: {exc}")
        return None
    return text.strip()[:1200] or None


class _JudgeModel:
    """Gives FixAgent's L1/L2 judge calls a longer decode budget than the
    short yes/no/option-letter evaluation calls need, without loading a
    second copy of the weights. ``FixAgent._ask_judge`` calls
    ``judge.generate(prompt)`` with no override, so with no wrapper the
    judge would inherit whatever ``--max-tokens`` was set for scoring
    (e.g. 8 for POPE) -- nowhere near enough to return a JSON candidate
    list, so it would silently truncate and fail to parse every time.

    320 (the original budget here) turned out to still be too small: it was
    measured directly against qwen3-vl-8b-instruct's actual L1/L2 judge
    outputs. The L1 prompt-template array is short and fits easily (~50
    output tokens observed), but the L2 pipeline-spec array (k proposals,
    each with image_ops/generation_kwargs/strategy/output_key_pattern) and
    the L2 code-writing prompt (~80-line Python pipeline) both got cut off
    mid-object/mid-function at 320, which is exactly what
    ``_ask_judge``/``_write_l2_coded``'s syntax gates report as "unparseable
    judge proposal" / "judge code failed to parse" -- a decode-budget bug,
    not a model-capability floor. 900 was measured to close both cleanly
    (full JSON array parsed, full syntactically-valid pipeline) on the same
    prompts that truncated at 320.
    """

    def __init__(self, model: HFLocalModel, max_new_tokens: int = 900) -> None:
        self._model = model
        self._max_new_tokens = max_new_tokens

    def generate(self, inputs: object, **kwargs: object) -> str:
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        return self._model.generate(inputs, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=PAPER_IDS)
    parser.add_argument("--model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--diagnosis-cases", type=int, default=24)
    parser.add_argument("--selection-cases", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--max-tier",
        default="L0",
        help="highest FixAgent intervention tier (for example L0 or L3a)",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-name")
    parser.add_argument("--data-dir", help="isolated local benchmark sample directory")
    parser.add_argument(
        "--exclude-data-dir",
        action="append",
        default=[],
        help="repeat for every observed local sample to exclude by content fingerprint",
    )
    parser.add_argument(
        "--allow-adapted-paper-methods",
        action="store_true",
        help="include methods marked architecture-adapted (never call their result an exact reproduction)",
    )
    parser.add_argument(
        "--ifcd-checkpoint",
        help=(
            "local TruthX checkpoint for the adapted IFCD route; keep it under "
            "the ignored data/ directory and record its provenance in the report"
        ),
    )
    parser.add_argument(
        "--only-paper-candidate",
        action="append",
        default=[],
        help="freeze validation to one or more pre-selected paper candidate names",
    )
    parser.add_argument(
        "--paper-prompt",
        action="store_true",
        help=(
            "use VCD's released POPE question-plus-one-word instruction instead "
            "of EvalVitals' answer-format prompt; supported for POPE conditions"
        ),
    )
    args = parser.parse_args()
    if args.paper_prompt and args.paper not in {"pope", "pope_popular", "pope_random"}:
        raise SystemExit("--paper-prompt is currently defined only for POPE conditions")
    unknown_candidates = set(args.only_paper_candidate) - PAPER_CANDIDATE_NAMES
    if unknown_candidates:
        choices = ", ".join(sorted(PAPER_CANDIDATE_NAMES))
        unknown = ", ".join(sorted(unknown_candidates))
        raise SystemExit(f"unknown paper candidate(s): {unknown}; choices: {choices}")
    if "ifcd_truthx_contrast" in args.only_paper_candidate and not args.ifcd_checkpoint:
        raise SystemExit("ifcd_truthx_contrast requires --ifcd-checkpoint")
    if args.ifcd_checkpoint and not Path(args.ifcd_checkpoint).is_file():
        raise SystemExit(f"--ifcd-checkpoint does not exist: {args.ifcd_checkpoint}")
    ifcd_checkpoint_sha256 = artifact_sha256(args.ifcd_checkpoint)
    split = args.diagnosis_cases + args.selection_cases
    if args.diagnosis_cases < 1 or args.selection_cases < 8 or split >= args.limit:
        raise SystemExit("need diagnosis >=1, selection >=8 and a non-empty confirmation split")

    source_dir = args.data_dir
    rows = (
        load_records_from(source_dir, args.paper)
        if source_dir is not None
        else load_records(args.paper)
    )
    rows, n_excluded = exclude_seen_rows(rows, args.paper, args.exclude_data_dir)
    if len(rows) < args.limit:
        raise SystemExit(
            f"only {len(rows)} unseen records remain after exclusion, fewer than --limit {args.limit}"
        )
    rows = shuffled_rows(rows[: args.limit], args.seed)
    probe_rows, confirm_rows = rows[:split], rows[split:]
    model = HFLocalModel(
        get_spec(args.model),
        RuntimeConfig(
            device=args.device,
            dtype="bfloat16",
            max_new_tokens=args.max_tokens,
            engine_kwargs={"ifcd_checkpoint": args.ifcd_checkpoint} if args.ifcd_checkpoint else {},
        ),
    )
    model.load()
    prompt_contract, prompt_fn = (
        paper_prompt_contract(args.only_paper_candidate)
        if args.paper_prompt
        else ("evalvitals_task_prompt", task_prompt)
    )
    vcd_sampling_control = bool(args.only_paper_candidate) and set(args.only_paper_candidate) <= {
        "vcd_diffusion_noise",
        "vcd_diffusion_noise_gated_false_yes",
    }

    def baseline_generate(row: dict[str, object]) -> str:
        model_inputs = Inputs(prompt_fn(row), row["image"])
        return (
            model.generate_vcd_baseline(model_inputs)
            if vcd_sampling_control
            else model.generate(model_inputs)
        )

    baseline_probe = evaluate(
        probe_rows, baseline_generate
    )
    diagnosis_rows, selection_rows = diagnostic_split(
        probe_rows, baseline_probe, args.diagnosis_cases
    )
    baseline_diagnosis = {
        "n": len(diagnosis_rows),
        "correct": sum(
            case["correct"]
            for case in baseline_probe["cases"]
            if case["id"] in {row["id"] for row in diagnosis_rows}
        ),
    }
    baseline_diagnosis["accuracy"] = baseline_diagnosis["correct"] / len(diagnosis_rows)
    baseline_selection = {
        "n": len(selection_rows),
        "cases": [
            case
            for case in baseline_probe["cases"]
            if case["id"] in {row["id"] for row in selection_rows}
        ],
    }
    baseline_selection["correct"] = sum(case["correct"] for case in baseline_selection["cases"])
    baseline_selection["accuracy"] = baseline_selection["correct"] / len(selection_rows)
    baseline_confirm = evaluate(confirm_rows, baseline_generate)

    # Default to L0 for decoding papers.  A caller can explicitly request L3a
    # for a read-only internal paper method such as ViCrop; the candidate
    # allowlist then freezes the paper route and prevents broad prompt/crop
    # screening from being mistaken for that method.
    #
    # The two local-visual-search papers keep hand-tuned statements: their
    # exact wording is what trips FixAgent's ``vicrop_mechanism`` keyword
    # gate (validated on mllms_know_textvqa_small, README +15.94pp). Do not
    # derive these from ``failure_axis`` — "small answer-region scene-text
    # perception" does not contain "resolution"/"small detail"/"tiny" and
    # would silently stop proposing ViCrop.
    local_visual_search_papers = {"mllms_know_textvqa_small", "vstar_bench"}
    diagnosis_text: str | None = None
    if args.paper in local_visual_search_papers:
        hypothesis_text = "Small answer-bearing visual details may be below the input resolution."
        predicted_failure_mode = "small visual detail"
    else:
        # Every other paper: derive the hypothesis from papers.json's own
        # ``failure_axis`` rather than a blanket "object hallucination" guess.
        # That default used to be fed to chartqa/mmmu_accounting too, which
        # mis-frames a chart-reading or expert-reasoning failure as language-
        # prior hallucination and steers L1/L2 candidate generation off-topic.
        failure_axis = PAPER_SPECS[args.paper].get(
            "failure_axis", "object hallucination and visual grounding"
        )
        hypothesis_text = f"Observed failures may trace to {failure_axis}."
        predicted_failure_mode = failure_axis
        # Real diagnosis (mirrors run_autofix.py's diagnose()): ask the model
        # itself to inspect actual diagnosis-split failures, rather than
        # relying only on the paper's pre-registered failure_axis. This is
        # diagnosis-split-only -- it never looks at selection/confirmation
        # cases -- so appending it to the hypothesis is not tuning.
        diagnosis_text = diagnose_hf(model, baseline_probe, diagnosis_rows)
        if diagnosis_text:
            hypothesis_text = f"{hypothesis_text} Model self-diagnosis: {diagnosis_text}"
    hypothesis = Hypothesis(
        statement=hypothesis_text,
        target_model=args.model,
        predicted_failure_mode=predicted_failure_mode,
        metadata={"fix_tier": args.max_tier},
    )
    # judge=_JudgeModel(model): without this, FixAgent's L1 (prompt) and L2
    # (scaffold) tiers never call an LLM at all -- _ask_judge returns []
    # immediately when judge is None -- so every unpinned run so far proposed
    # only the fixed deterministic defaults (visual_grounding /
    # salient_crop / upscale_sharpen / zoom_equalize) regardless of what the
    # diagnosis said. run_autofix.py's black-box runner already does this
    # (judge=model); the HF runner never did.
    #
    # Measured, not assumed: on llava-1.5-7b-hf this is currently a no-op.
    # A smoke test showed the judge's JSON proposals fail to parse (falls
    # back to the same defaults, logged as a warning) and its free-text
    # self-diagnosis answered the embedded question instead of doing the
    # meta-task ("The image does not show a bicycle." instead of a failure
    # mechanism) -- a capability floor of the 7B subject model, not a prompt
    # or wiring bug. Left in rather than reverted because it is correct
    # infrastructure for a stronger local judge (e.g. a Qwen-VL judge
    # instance decoupled from the model under test); harmless overhead on a
    # model too weak to use it.
    agent = FixAgent(
        judge=_JudgeModel(model),
        max_tier=args.max_tier,
        score_fn=score_case,
        max_validation_cases=0,
        allow_adapted_paper_methods=args.allow_adapted_paper_methods,
        candidate_allowlist=args.only_paper_candidate or None,
    )
    selection = agent.propose_and_validate(
        model,
        make_cases(selection_rows, baseline_selection, prompt_fn=prompt_fn),
        [hypothesis],
    )
    confirmation: dict[str, object] = {"skipped": "no selection candidate"}
    if selection.best is not None:
        validated = FixAgent(score_fn=score_case).validate_candidate(
            model,
            make_cases(confirm_rows, baseline_confirm, prompt_fn=prompt_fn),
            selection.best.candidate,
        )
        confirmation = {
            "candidate": selection.best.candidate.name,
            "fixed": validated.fixed,
            "n_fixed": validated.n_fixed,
            "n_broken": validated.n_broken,
            "effect": validated.effect,
            "e_value": validated.e_value,
            "baseline_accuracy": validated.n_baseline_correct / validated.n_pairs,
            "candidate_accuracy": validated.n_candidate_correct / validated.n_pairs,
            "summary": validated.summary,
        }
    report = {
        "paper": args.paper,
        "backend": "hf_local",
        "model": args.model,
        "method": "paper-method candidates over HF-local internals/decoding",
        "max_tier": args.max_tier,
        "prompt_contract": prompt_contract,
        "baseline_decoding": "vcd_clean_temperature_one_sampling" if vcd_sampling_control else "default_generate",
        "allow_adapted_paper_methods": args.allow_adapted_paper_methods,
        "model_self_diagnosis": diagnosis_text,
        "only_paper_candidates": args.only_paper_candidate,
        "ifcd_checkpoint": args.ifcd_checkpoint,
        "ifcd_checkpoint_sha256": ifcd_checkpoint_sha256,
        "data_dir": source_dir,
        "exclude_data_dirs": args.exclude_data_dir,
        "n_excluded_as_previously_seen": n_excluded,
        "splits": {
            "diagnosis": len(diagnosis_rows),
            "selection": len(selection_rows),
            "confirmation": len(confirm_rows),
            "shuffle_seed": args.seed,
        },
        "baseline": {
            "diagnosis": baseline_diagnosis,
            "selection": baseline_selection,
            "confirmation": baseline_confirm,
        },
        "hypothesis": hypothesis_to_dict(hypothesis),
        "auto_fix": {"selection": selection.to_dict(), "confirmation": confirmation},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.output_name or args.paper + '_hf'}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "confirmation": confirmation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
