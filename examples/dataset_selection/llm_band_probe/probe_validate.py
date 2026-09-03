"""Run the 2026-08 reasoning probes against a real endpoint.

The probes so far have only been exercised by scripted models, which proves the
columns compute but not that they *discriminate* on real generations.  This
script collects a baseline run on one dataset slice (so PASS/FAIL labels are
real), then runs the probes in the order they are meant to be read:

    hygiene (extraction, termination)  ->  free mechanism (arith)
      ->  interventional mechanism (coverage, perturbation, self-repair, cot)

and reports the columns side by side.  Reading the hygiene probes first is not
a style preference: if the FAIL labels are partly parse failures or truncations,
every mechanism number below them is measured on a contaminated pool.

    python probe_validate.py --dataset gsm_symbolic_p2 --n 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import band_locate as B  # noqa: E402

from evalrx.analyzers.perturbation.cot_faithfulness import CoTFaithfulnessAnalyzer  # noqa: E402
from evalrx.analyzers.perturbation.perturbation_battery import PerturbationBattery  # noqa: E402
from evalrx.analyzers.reasoning.answer_extraction_audit import (
    AnswerExtractionAudit,  # noqa: E402
)
from evalrx.analyzers.reasoning.arith_audit import ArithmeticAudit  # noqa: E402
from evalrx.analyzers.reasoning.self_repair import SelfRepairAnalyzer  # noqa: E402
from evalrx.analyzers.reasoning.termination_audit import TerminationAudit  # noqa: E402
from evalrx.analyzers.uncertainty.coverage_gap import CoverageVerificationGap  # noqa: E402
from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer  # noqa: E402
from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.model import Model  # noqa: E402


class EndpointModel(Model):
    """Minimal OpenAI-compatible chat model for the probes."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, max_tokens: int = 3072,
                 temperature: float = B.SAMPLING["temperature"],
                 sampling: dict | None = None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Greedy decoding sends this model into verbatim self-verification loops
        # that run to the token cap; every probe reading such a chain would be
        # measuring the decoding config. See SAMPLING in band_locate.
        self.sampling = (
            {k: v for k, v in B.SAMPLING.items() if k != "temperature"}
            if sampling is None else sampling
        )
        self.n_calls = 0
        self.n_truncated = 0

    def generate(self, inputs, **kwargs):
        self.n_calls += 1
        text, reason = B.generate(
            str(inputs.prompt),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("temperature", self.temperature),
            sampling=self.sampling,
        )
        if reason == "length":
            self.n_truncated += 1
        return text

    def logprobs(self, inputs, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"EndpointModel({B.MODEL_ID})"


def collect_baseline(spec: B.Spec, n: int, concurrency: int, max_tokens: int) -> CaseBatch:
    """Run the model once per item and build a LABELLED batch from the outcome."""
    rows = B.fetch_rows(spec, n)
    adapter = spec.adapter or B._adapter_plain(spec.question_field, spec.answer_field)
    items = []
    for row in rows:
        pair = adapter(row)
        if pair:
            items.append(pair)
        if len(items) >= n:
            break

    grade = spec.grader or B.answer_equal

    sampling = {k: v for k, v in B.SAMPLING.items() if k != "temperature"}

    def _one(item):
        question, gold = item
        prompt = (
            f"{question}\n\n{spec.instruction}" if spec.append_instruction
            else question
        )
        output, _ = B.generate(
            prompt, max_tokens, B.SAMPLING["temperature"], sampling=sampling
        )
        # a spec whose gold spans lines is graded on the whole generation, the
        # same way band_locate grades it — otherwise PASS/FAIL here would
        # disagree with the band the dataset was chosen on
        graded_text = output if spec.grades_raw_output else B.extract_answer(output)
        ok = bool(grade(graded_text, gold))
        return FailureCase(
            inputs=Inputs(prompt=prompt),
            observed=output,
            expected=gold if not isinstance(gold, (list, tuple)) else gold[0],
            label=Label.PASS if ok else Label.FAIL,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        cases = list(pool.map(_one, items))
    return CaseBatch(cases)


def summarise(findings: dict, keys: tuple) -> dict:
    return {k: findings.get(k) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm_symbolic_p2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--out", default="probe_validation.json")
    ap.add_argument("--probe-cases", type=int, default=12,
                    help="cap for the INTERVENTIONAL probes. Their model calls are "
                         "SEQUENTIAL inside the analyzer, so against a local "
                         "endpoint this is the wall-clock knob that matters: at "
                         "~130s per long generation, k=5 over 12 cases is 2+ hours "
                         "for one probe.")
    ap.add_argument("--only", default="",
                    help="comma-separated probe names to run (default: all)")
    args = ap.parse_args()

    spec = next(s for s in B.SPECS if s.name == args.dataset)
    print(f"[baseline] {spec.name} n={args.n}", flush=True)
    started = time.time()
    batch = collect_baseline(spec, args.n, args.concurrency, args.max_tokens)
    n_pass = sum(1 for c in batch if c.label == Label.PASS)
    print(
        f"  baseline accuracy {n_pass}/{len(batch)} = {n_pass / max(len(batch), 1):.3f} "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )

    model = EndpointModel(max_tokens=args.max_tokens)
    report: dict = {
        "model": B.MODEL_ID,
        "dataset": spec.name,
        "n_cases": len(batch),
        "baseline_accuracy": round(n_pass / max(len(batch), 1), 4),
        "probes": {},
    }

    # Hygiene and free probes first — they decide whether the rest is readable.
    plan = [
        ("answer_extraction_audit", AnswerExtractionAudit(),
         ("n_gradable", "n_labelled_fail", "n_extraction_suspect", "suspect_rate",
          "missing_tag_rate", "n_extraction_point_miss")),
        ("termination_audit", TerminationAudit(max_cases=args.n),
         ("class_counts", "clean_rate", "truncation_rate", "degenerate_rate",
          "giveup_rate", "recovered_rate")),
        ("arith_audit", ArithmeticAudit(),
         ("n_with_equations", "n_wrong_answers", "mean_arith_error_rate",
          "computation_slip_rate", "chain_break_rate", "compound_rate",
          "n_errors_but_correct")),
        ("coverage_verification_gap",
         CoverageVerificationGap(k=5, max_cases=args.probe_cases,
                                 gen_kwargs={"temperature": 0.8}),
         ("n_scored", "mean_pass_at_k", "mean_majority_correct", "coverage_gap_rate",
          "no_coverage_rate", "degenerate_sampling")),
        ("perturbation_battery", PerturbationBattery(max_cases=args.probe_cases),
         ("n_scored", "applied_perturbations", "mean_invariance_break_rate",
          "noop_break_rate", "mean_sensitivity_rate", "n_memorization_suspect")),
        ("self_repair", SelfRepairAnalyzer(max_cases=args.probe_cases),
         ("n_baseline_fail", "n_baseline_pass", "detection_accuracy",
          "false_alarm_rate", "repair_rate", "damage_rate", "net_revision_gain")),
        ("cot_faithfulness", CoTFaithfulnessAnalyzer(max_cases=args.probe_cases),
         ("mean_early_match_rate", "mean_cot_effect", "n_graded", "drift_away_rate",
          "late_rescue_rate", "mean_first_correct_frac", "mean_wasted_reasoning_frac")),
        ("self_consistency", SelfConsistencyAnalyzer(n=5, gen_kwargs={"temperature": 0.8}),
         ("consistency", "n_unique", "semantic_consistency", "n_semantic_clusters",
          "semantic_entropy", "normalized_semantic_entropy")),
    ]

    wanted = {p.strip() for p in args.only.split(",") if p.strip()}
    if wanted:
        plan = [entry for entry in plan if entry[0] in wanted]
    for name, analyzer, keys in plan:
        t0, calls0 = time.time(), model.n_calls
        try:
            findings = analyzer.run(model, batch).findings
        except Exception as exc:
            print(f"[{name}] ERROR {type(exc).__name__}: {exc}", flush=True)
            report["probes"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        summary = summarise(findings, keys)
        summary["_seconds"] = round(time.time() - t0, 1)
        summary["_model_calls"] = model.n_calls - calls0
        report["probes"][name] = summary
        print(f"[{name}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
