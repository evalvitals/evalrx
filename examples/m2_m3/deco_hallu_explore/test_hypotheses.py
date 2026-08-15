"""Pipeline phase 2 — held-out hypothesis testing.

Takes phase 1's exploratory report (candidate signals with FROZEN recipes +
M3 hypotheses with test_design) and evaluates them on the validate split the
explorer never saw:

1. Each candidate recipe's expression — thresholds exactly as fitted in
   phase 1, NO re-fitting — is compiled over the validate rows
   (`analysis.operationalize.compile_recipe`), rebuilt into two-group
   sufficient statistics, and adjudicated by the host
   (`analysis.adjudicate.adjudicate_signals`, split_label="held_out").
   Unlike phase 1's in-sample screen, a REJECT here is a real held-out verdict.
2. An LLM judge reads each M3 hypothesis (statement + test_design) next to the
   held-out statistics and grades it supported / partial / refuted /
   not_testable — with reasoning. Hypotheses whose mechanism cannot be decided
   from observational held-out stats are marked not_testable and routed to
   phase 3 (surgery) rather than silently "supported".

Writes <pipeline-root>/1_explore/confirm_report.json (next to the exploratory
report, where the dashboard picks it up).

    python test_hypotheses.py --pipeline-root outputs_pipeline
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

JUDGE_PROMPT = """You are adjudicating a proposed hypothesis about VLM hallucination
failures, using HELD-OUT data the proposer never saw.

HYPOTHESIS: {statement}
BASIS (from the exploration phase): {basis}
PROPOSED TEST DESIGN: {test_design}

HELD-OUT EVIDENCE (validate split, n={n_rows}; thresholds frozen from the
exploration phase; REJECT H0 means the signal separated FAIL/PASS on held-out
data):
{evidence}

Grade the hypothesis STRICTLY against this held-out evidence:
- "supported": the held-out statistics directly back the hypothesis's claim.
- "partial": the correlational part holds up but the hypothesis claims more
  (e.g. a mechanism or causal direction) than these statistics can establish.
- "refuted": the held-out statistics contradict the claim.
- "not_testable": these observational statistics cannot decide the claim at
  all (e.g. causal direction); it needs an intervention/surgery experiment.

Reply with ONLY a JSON object:
{{"verdict": "supported|partial|refuted|not_testable",
  "reasoning": "<2-3 sentences citing the held-out numbers>",
  "needs_surgery": true/false}}
"""


def _load_validate_rows(validate_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(validate_dir.glob("*.json")):
        raw = json.loads(f.read_text())
        model = str(raw.get("model", f.stem))
        for r in raw["cases"]:
            row = dict(r)
            row["model"] = model
            row["case_id"] = f"{model}:{r.get('image_id')}:{r.get('object')}"
            row["is_fail"] = 1.0 if r.get("label") == "fail" else 0.0
            rows.append(row)
    return rows


def _holdout_sufficient(recipe_values: dict[str, float], rows: list[dict]) -> dict | None:
    """Rebuild two-group sufficient stats on held-out rows: is_fail indicators
    among signal-ABSENT (a) vs signal-PRESENT (b) cases. Only defined for
    recipes that evaluate to a binary flag (threshold recipes do)."""
    by_id = {r["case_id"]: r for r in rows}
    vals = {cid: v for cid, v in recipe_values.items() if cid in by_id}
    if not vals:
        return None
    uniq = sorted({round(v, 6) for v in vals.values()})
    if not set(uniq) <= {0.0, 1.0}:
        return None  # continuous recipe — no frozen threshold to reuse
    a = [int(by_id[cid]["is_fail"]) for cid, v in vals.items() if v == 0.0]
    b = [int(by_id[cid]["is_fail"]) for cid, v in vals.items() if v == 1.0]
    if not a or not b:
        return None
    return {"kind": "two_group", "a": a, "b": b}


def _judge_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", default="outputs_pipeline")
    ap.add_argument("--validate-dir", default=str(HERE / "data_attn_validate"))
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--judge-effort", default="low")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM judge (signal verdicts only)")
    args = ap.parse_args()

    root = Path(args.pipeline_root)
    explore_dir = root / "1_explore"
    report_path = explore_dir / "exploratory_report.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} missing — run phase 1 (evalvitals explore) first")
    report = json.loads(report_path.read_text())
    candidates_raw = [c for c in report.get("candidate_signals") or [] if isinstance(c, dict)]
    hypotheses = [h for h in report.get("hypotheses") or [] if isinstance(h, dict)]

    rows = _load_validate_rows(Path(args.validate_dir))
    if not rows:
        raise SystemExit(f"no validate rows under {args.validate_dir} — run prepare_splits.py")
    n_fail = sum(int(r["is_fail"]) for r in rows)
    print(f"held-out validate rows: {len(rows)} ({n_fail} FAIL)")

    from evalvitals.analysis.adjudicate import adjudicate_signals
    from evalvitals.analysis.explorer import CandidateSignal
    from evalvitals.analysis.operationalize import RecipeError, SignalRecipe, compile_recipe

    # ── 1. frozen-recipe re-evaluation on the held-out rows ────────────────
    signal_verdicts: list[dict] = []
    candidates: list[CandidateSignal] = []
    for c in candidates_raw:
        entry = {
            "name": c.get("name"),
            "display_name": c.get("display_name") or c.get("name"),
            "recipe": c.get("recipe"),
            "explore_effect": c.get("effect"),
            "explore_reject_in_sample": c.get("reject"),
        }
        recipe_dict = c.get("recipe")
        if not isinstance(recipe_dict, dict) or recipe_dict.get("kind") != "expr":
            entry.update(status="skipped", reason="no expr recipe to re-evaluate")
            signal_verdicts.append(entry)
            continue
        try:
            values = compile_recipe(SignalRecipe.from_dict(recipe_dict), rows)
        except (RecipeError, NotImplementedError) as exc:
            entry.update(status="skipped", reason=f"recipe error: {exc}")
            signal_verdicts.append(entry)
            continue
        sufficient = _holdout_sufficient(values, rows)
        if sufficient is None:
            entry.update(status="skipped",
                         reason="recipe is not a binary flag on held-out rows "
                                "(no frozen threshold to reuse)")
            signal_verdicts.append(entry)
            continue
        cand = CandidateSignal(name=str(c.get("name") or "signal"),
                               rationale=str(c.get("rationale") or ""),
                               recipe=recipe_dict, sufficient=sufficient)
        entry.update(status="adjudicated",
                     n_holdout=len(sufficient["a"]) + len(sufficient["b"]),
                     n_flagged=len(sufficient["b"]),
                     fail_rate_flagged=round(sum(sufficient["b"]) / max(1, len(sufficient["b"])), 4),
                     fail_rate_unflagged=round(sum(sufficient["a"]) / max(1, len(sufficient["a"])), 4))
        signal_verdicts.append(entry)
        candidates.append(cand)

    meta = adjudicate_signals(candidates, alpha=args.alpha, split_label="held_out")
    by_name: dict[str, CandidateSignal] = {c.name: c for c in candidates}
    for entry in signal_verdicts:
        cand = by_name.get(str(entry.get("name")))
        if entry.get("status") == "adjudicated" and cand is not None:
            entry.update(reject=bool(cand.reject),
                         effect=cand.effect, ci=cand.ci,
                         fdr_corrected=bool(cand.fdr_corrected))
    print(f"held-out adjudication: {meta['n_rejected']}/{meta['n_host_adjudicated']} "
          f"reject (alpha={args.alpha})")
    for e in signal_verdicts:
        tag = ("REJECT H0" if e.get("reject") else "no reject") \
            if e.get("status") == "adjudicated" else e.get("reason")
        print(f" - {e.get('name')}: {tag}")

    # ── 2. LLM judge per hypothesis, grounded in the held-out table ────────
    evidence_lines = [
        f"- {e.get('name')}: flagged-group fail rate {e.get('fail_rate_flagged')} "
        f"vs unflagged {e.get('fail_rate_unflagged')} "
        f"(n={e.get('n_holdout')}, effect={e.get('effect')}, CI={e.get('ci')}, "
        f"verdict={'REJECT H0' if e.get('reject') else 'not rejected'})"
        for e in signal_verdicts if e.get("status") == "adjudicated"
    ] or ["- (no recipe could be re-evaluated on the held-out split)"]
    evidence = "\n".join(evidence_lines)

    hypothesis_verdicts: list[dict] = []
    if hypotheses and not args.no_judge:
        from evalvitals.eval_agent import ClaudeModel

        judge = ClaudeModel(model=args.judge_model, effort=args.judge_effort)
        for h in hypotheses:
            prompt = JUDGE_PROMPT.format(
                statement=h.get("statement", ""), basis=h.get("basis", ""),
                test_design=h.get("test_design", ""), n_rows=len(rows),
                evidence=evidence)
            parsed = _judge_json(judge.generate(prompt) or "")
            verdict = dict(h)
            if parsed and parsed.get("verdict") in {"supported", "partial", "refuted", "not_testable"}:
                verdict.update(verdict=parsed["verdict"],
                               reasoning=str(parsed.get("reasoning", "")),
                               needs_surgery=bool(parsed.get("needs_surgery", False)))
            else:
                verdict.update(verdict="not_testable",
                               reasoning="judge output unparseable — defaulting to not_testable",
                               needs_surgery=True)
            hypothesis_verdicts.append(verdict)
            print(f" * [{verdict['verdict']}] {verdict['statement'][:90]}")
    else:
        for h in hypotheses:
            hypothesis_verdicts.append({**h, "verdict": "not_judged",
                                        "reasoning": "", "needs_surgery": True})

    out = {
        "phase": "holdout_confirm",
        "split": "held_out",
        "n_validate_rows": len(rows),
        "n_validate_fail": n_fail,
        "alpha": args.alpha,
        "adjudication": meta,
        "signal_verdicts": signal_verdicts,
        "hypothesis_verdicts": hypothesis_verdicts,
        "judge": None if args.no_judge else {"model": args.judge_model,
                                             "effort": args.judge_effort},
    }
    out_path = explore_dir / "confirm_report.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
