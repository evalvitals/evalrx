"""Generic held-out verification for standalone explore runs.

Generalizes the deco_hallu_explore example's ``test_hypotheses.py`` to ANY
dataset the explorer can load (the upload workbench's mode 2, and
``evalrx explore --holdout-frac F --holdout-confirm``):

1. :func:`split_records` carves the loaded records into a deterministic,
   outcome-stratified explore/holdout partition BEFORE exploration — the
   explorer only ever sees the explore share.
2. :func:`holdout_confirm` re-evaluates the explorer's FROZEN recipes verbatim
   on the held-out rows (:func:`~evalrx.analysis.operationalize.compile_recipe`,
   thresholds exactly as fitted — no re-fitting), rebuilds two-group sufficient
   statistics, adjudicates with
   :func:`~evalrx.analysis.adjudicate.adjudicate_signals`
   (``split_label="held_out"`` — a REJECT here is a real held-out verdict), and
   optionally has an LLM judge grade each M3 hypothesis against the held-out
   table (supported / partial / refuted / not_testable + surgery routing).

The returned dict is the ``confirm_report.json`` shape the dashboard's
*Held-out Verdicts* tab renders. The judge is duck-typed (any object with
``generate(str) -> str``); pass ``judge=None`` for signal verdicts only —
hypotheses then carry ``verdict="not_judged"``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalrx.analysis.adjudicate import adjudicate_signals
from evalrx.analysis.explorer import CandidateSignal
from evalrx.analysis.fused_pipeline import _split_records
from evalrx.analysis.operationalize import RecipeError, SignalRecipe, compile_recipe
from evalrx.analysis.prompts.holdout import JUDGE_PROMPT

_FAIL_LIKE = ("fail", "failed", "failure", "error", "wrong", "incorrect", "no")


def split_records(
    records: "list[dict[str, Any]]",
    holdout_frac: float,
    *,
    seed: int = 0,
    outcome_col: str | None = None,
) -> "tuple[list[dict[str, Any]], list[dict[str, Any]] | None]":
    """Deterministic ``(explore, holdout)`` partition, stratified by outcome.

    Wraps the fused pipeline's split so the standalone path and the loop's
    fused path hold out rows the same way. Returns ``(records, None)`` when
    the fraction or batch size makes a split meaningless — callers fall back
    to a plain in-sample run with a caveat rather than a fake split.
    """
    return _split_records(
        records, frac=holdout_frac, seed=seed, label_col=outcome_col or "label"
    )


def failure_indicator(
    rows: "list[dict[str, Any]]", outcome_col: str
) -> "tuple[list[int | None], Any, str]":
    """Per-row binary failure indicator for an arbitrary outcome column.

    Returns ``(indicators, positive_label, note)`` where ``indicators[i]`` is
    1 (failure), 0 (non-failure) or None (missing outcome). The positive
    ("failure") class is chosen by, in order: a fail-like string value
    (fail/failed/error/wrong/...), the truthy pole of a boolean/0-1 column, or
    — for any other two-valued column — the MINORITY value (failures are
    normally the rare class); *note* says which rule fired so the report can
    state it honestly.
    """
    values = [r.get(outcome_col) for r in rows]
    present = [v for v in values if v is not None]
    uniq = {str(v).strip().lower() for v in present}

    positive: Any = None
    note = ""
    fail_hits = sorted(uniq & set(_FAIL_LIKE))
    if fail_hits:
        positive = fail_hits[0]
        note = f"failure = {outcome_col!r} == {positive!r} (fail-like value)"
    elif uniq <= {"0", "1", "true", "false", "0.0", "1.0"} and uniq:
        positive = "truthy"
        note = f"failure = truthy {outcome_col!r} (boolean/0-1 column)"
    elif len(uniq) == 2:
        counts: dict[str, int] = {}
        for v in present:
            key = str(v).strip().lower()
            counts[key] = counts.get(key, 0) + 1
        positive = min(counts, key=lambda k: counts[k])
        note = (
            f"failure = {outcome_col!r} == {positive!r} "
            "(minority value of a two-valued column)"
        )
    else:
        return [None] * len(rows), None, (
            f"outcome column {outcome_col!r} is not binary "
            f"({len(uniq)} distinct values) — held-out verdicts unavailable"
        )

    out: "list[int | None]" = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        key = str(v).strip().lower()
        if positive == "truthy":
            out.append(1 if key in {"1", "true", "1.0"} else 0)
        else:
            out.append(1 if key == positive else 0)
    return out, positive, note


def _holdout_sufficient(
    values: "dict[str, float]",
    id_to_fail: "dict[str, int]",
) -> "dict[str, Any] | None":
    """Two-group sufficient stats: failure indicators among signal-ABSENT (a)
    vs signal-PRESENT (b) rows. Defined only for binary-flag recipes — a
    continuous recipe has no frozen threshold to reuse."""
    pairs = [(v, id_to_fail[cid]) for cid, v in values.items() if cid in id_to_fail]
    if not pairs:
        return None
    if not {round(v, 6) for v, _ in pairs} <= {0.0, 1.0}:
        return None
    a = [f for v, f in pairs if v == 0.0]
    b = [f for v, f in pairs if v == 1.0]
    if not a or not b:
        return None
    return {"kind": "two_group", "a": a, "b": b}


def _judge_json(text: str) -> "dict[str, Any] | None":
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def holdout_confirm(
    report: "dict[str, Any]",
    holdout_rows: "list[dict[str, Any]]",
    *,
    outcome_col: str = "label",
    alpha: float = 0.05,
    judge: Any = None,
    judge_meta: "dict[str, Any] | None" = None,
) -> "dict[str, Any]":
    """Re-test one exploratory report's recipes + hypotheses on held-out rows.

    *report* is the ``exploratory_report.json`` dict (``to_dict()`` of a live
    report works too). Returns the ``confirm_report.json`` payload; the caller
    persists it next to the exploratory report.
    """
    rows = [dict(r) for r in holdout_rows]
    indicators, positive_label, outcome_note = failure_indicator(rows, outcome_col)

    # Stable ids: force the compile id_col so recipe values map back to rows
    # regardless of what id-ish keys the upload carries.
    id_to_fail: "dict[str, int]" = {}
    for i, (row, ind) in enumerate(zip(rows, indicators)):
        row["_holdout_id"] = f"h{i}"
        if ind is not None:
            id_to_fail[f"h{i}"] = ind
    n_fail = sum(id_to_fail.values())

    candidates_raw = [c for c in report.get("candidate_signals") or [] if isinstance(c, dict)]
    hypotheses = [h for h in report.get("hypotheses") or [] if isinstance(h, dict)]

    # ── 1. frozen-recipe re-evaluation on the held-out rows ─────────────────
    signal_verdicts: "list[dict[str, Any]]" = []
    candidates: "list[CandidateSignal]" = []
    for c in candidates_raw:
        entry: "dict[str, Any]" = {
            "name": c.get("name"),
            "display_name": c.get("display_name") or c.get("name"),
            "recipe": c.get("recipe"),
            "explore_effect": c.get("effect"),
            "explore_reject_in_sample": c.get("reject"),
        }
        recipe_dict = c.get("recipe")
        if not id_to_fail:
            entry.update(status="skipped", reason=outcome_note or "no binary outcome")
            signal_verdicts.append(entry)
            continue
        if not isinstance(recipe_dict, dict) or recipe_dict.get("kind") != "expr":
            entry.update(status="skipped", reason="no expr recipe to re-evaluate")
            signal_verdicts.append(entry)
            continue
        try:
            values = compile_recipe(
                SignalRecipe.from_dict(recipe_dict), rows, id_col="_holdout_id"
            )
        except (RecipeError, NotImplementedError) as exc:
            entry.update(status="skipped", reason=f"recipe error: {exc}")
            signal_verdicts.append(entry)
            continue
        sufficient = _holdout_sufficient(values, id_to_fail)
        if sufficient is None:
            entry.update(
                status="skipped",
                reason="recipe is not a binary flag on held-out rows "
                       "(no frozen threshold to reuse)",
            )
            signal_verdicts.append(entry)
            continue
        cand = CandidateSignal(
            name=str(c.get("name") or "signal"),
            rationale=str(c.get("rationale") or ""),
            recipe=recipe_dict,
            sufficient=sufficient,
        )
        n_a, n_b = len(sufficient["a"]), len(sufficient["b"])
        entry.update(
            status="adjudicated",
            n_holdout=n_a + n_b,
            n_flagged=n_b,
            fail_rate_flagged=round(sum(sufficient["b"]) / n_b, 4),
            fail_rate_unflagged=round(sum(sufficient["a"]) / n_a, 4),
        )
        signal_verdicts.append(entry)
        candidates.append(cand)

    meta = adjudicate_signals(candidates, alpha=alpha, split_label="held_out")
    by_name = {c.name: c for c in candidates}
    for entry in signal_verdicts:
        cand = by_name.get(str(entry.get("name")))
        if entry.get("status") == "adjudicated" and cand is not None:
            entry.update(
                reject=bool(cand.reject),
                effect=cand.effect,
                ci=cand.ci,
                fdr_corrected=bool(cand.fdr_corrected),
            )

    # ── 2. LLM judge per hypothesis, grounded in the held-out table ─────────
    evidence_lines = [
        f"- {e.get('name')}: flagged-group fail rate {e.get('fail_rate_flagged')} "
        f"vs unflagged {e.get('fail_rate_unflagged')} "
        f"(n={e.get('n_holdout')}, effect={e.get('effect')}, CI={e.get('ci')}, "
        f"verdict={'REJECT H0' if e.get('reject') else 'not rejected'})"
        for e in signal_verdicts
        if e.get("status") == "adjudicated"
    ] or ["- (no recipe could be re-evaluated on the held-out split)"]
    evidence = "\n".join(evidence_lines)

    hypothesis_verdicts: "list[dict[str, Any]]" = []
    for h in hypotheses:
        verdict = dict(h)
        if judge is None:
            verdict.update(verdict="not_judged", reasoning="", needs_surgery=True)
            hypothesis_verdicts.append(verdict)
            continue
        prompt = JUDGE_PROMPT.format(
            statement=h.get("statement", ""),
            basis=h.get("basis", ""),
            test_design=h.get("test_design", ""),
            n_rows=len(rows),
            evidence=evidence,
        )
        try:
            raw = judge.generate(prompt) or ""
        except Exception as exc:  # noqa: BLE001 — judge outage must not sink the verdicts
            raw = ""
            verdict["judge_error"] = str(exc)
        parsed = _judge_json(raw)
        if parsed and parsed.get("verdict") in {"supported", "partial", "refuted", "not_testable"}:
            verdict.update(
                verdict=parsed["verdict"],
                reasoning=str(parsed.get("reasoning", "")),
                needs_surgery=bool(parsed.get("needs_surgery", False)),
            )
        else:
            verdict.update(
                verdict="not_testable",
                reasoning="judge output unparseable — defaulting to not_testable",
                needs_surgery=True,
            )
        hypothesis_verdicts.append(verdict)

    return {
        "phase": "holdout_confirm",
        "split": "held_out",
        "n_validate_rows": len(rows),
        "n_validate_fail": n_fail,
        "alpha": alpha,
        "outcome": {
            "column": outcome_col,
            "positive_label": positive_label,
            "note": outcome_note,
        },
        "adjudication": meta,
        "signal_verdicts": signal_verdicts,
        "hypothesis_verdicts": hypothesis_verdicts,
        "judge": judge_meta if judge is not None else None,
    }
