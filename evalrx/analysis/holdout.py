"""Generic held-out verification for standalone explore runs.

Generalizes the deco_hallu_explore example's ``test_hypotheses.py`` to ANY
dataset the explorer can load (the upload workbench's mode 2, and
``evalrx explore --holdout-frac F --holdout-confirm``):

1. :func:`split_records` carves the loaded records into a deterministic,
   outcome-stratified explore/holdout partition BEFORE exploration — the
   explorer only ever sees the explore share. With a group column, rows that
   share a group stay on one side.
2. :func:`holdout_confirm` re-evaluates the explorer's FROZEN recipes verbatim
   on the held-out rows (thresholds, subgroup conditions, strata and cut
   points exactly as fitted — no re-fitting), estimates each one's effect with
   a cluster bootstrap (:mod:`~evalrx.analysis.holdout_designs`), corrects all
   claims in one Benjamini-Hochberg family, checks the numeric predictions a
   hypothesis carries, and optionally has an LLM judge grade each M3
   hypothesis against the held-out table (supported / partial / refuted /
   not_testable + surgery routing).

The returned dict is the ``confirm_report.json`` shape the dashboard's
*Held-out Verdicts* tab renders. The judge is duck-typed (any object with
``generate(str) -> str``); pass ``judge=None`` for signal verdicts only —
hypotheses then carry ``verdict="not_judged"`` and, when they stated numeric
predictions, a ``host_verdict`` that needed no judge.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from evalrx.agent_runtime.judge_call import JudgeError, ask_judge
from evalrx.analysis.fused_pipeline import _split_records
from evalrx.analysis.holdout_designs import (
    DesignEstimate,
    claim_status,
    contrary_p_value,
    estimate_design,
    representative_rows,
)
from evalrx.analysis.hypothesis_agent import ExperimentPlan
from evalrx.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    expr_columns,
    parse_expectation,
)
from evalrx.analysis.prompts.holdout import JUDGE_PROMPT
from evalrx.stats.bootstrap import bootstrap_p_value
from evalrx.stats.multiplicity import bh

_FAIL_LIKE = ("fail", "failed", "failure", "error", "wrong", "incorrect", "no")


def split_records(
    records: "list[dict[str, Any]]",
    holdout_frac: float,
    *,
    seed: int = 0,
    outcome_col: str | None = None,
    group_col: str | None = None,
) -> "tuple[list[dict[str, Any]], list[dict[str, Any]] | None]":
    """Deterministic ``(explore, holdout)`` partition, stratified by outcome.

    Wraps the fused pipeline's split so the standalone path and the loop's
    fused path hold out rows the same way. Returns ``(records, None)`` when
    the fraction or batch size makes a split meaningless — callers fall back
    to a plain in-sample run with a caveat rather than a fake split.

    *group_col* names the unit rows were cut from; rows that share its value
    all go to the same side, so the held-out half holds no near-copy of a row
    the explorer saw.
    """
    return _split_records(
        records, frac=holdout_frac, seed=seed, label_col=outcome_col or "label",
        group_col=group_col,
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


def outcome_values(
    rows: "list[dict[str, Any]]", outcome_col: str
) -> "tuple[list[float | None], str | None, Any, str]":
    """The outcome as one number per row: ``(values, kind, positive_label, note)``.

    ``kind`` is ``"binary"`` (values are 0/1 failure indicators, chosen by
    :func:`failure_indicator`), ``"continuous"`` (a numeric outcome used as it
    is, so effects are differences of means), or None when the column is
    neither — then nothing can be re-tested and *note* says why.
    """
    indicators, positive, note = failure_indicator(rows, outcome_col)
    if positive is not None:
        return [None if v is None else float(v) for v in indicators], "binary", positive, note
    present = [r.get(outcome_col) for r in rows if r.get(outcome_col) is not None]
    if present and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in present
    ):
        values = [
            None if r.get(outcome_col) is None else float(r[outcome_col]) for r in rows
        ]
        return values, "continuous", None, (
            f"outcome = numeric {outcome_col!r}; effects are differences of its mean "
            "(rows a recipe flags minus rows it does not)"
        )
    return [None] * len(rows), None, None, note


class _HoldoutVerdict(BaseModel):
    verdict: Literal["supported", "partial", "refuted", "not_testable"]
    reasoning: str
    needs_surgery: bool = False
    # The plan for the intervention, when one is needed and the hypothesis
    # did not already carry one.
    experiment: ExperimentPlan | None = None


_ID_KEYS = ("case_id", "id", "sample_id")


def _row_ref(row: "dict[str, Any]", index: int, id_col: str | None) -> "dict[str, Any]":
    """A pointer to one held-out row: its position and, when present, its id."""
    for key in ((id_col,) if id_col else ()) + _ID_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return {"row": index, "id": value}
    return {"row": index, "id": None}


def _evidence_line(entry: "dict[str, Any]", outcome_kind: str | None) -> str:
    """One line of the held-out table a judge reads, per adjudicated signal."""
    name = entry.get("name")
    claim = entry.get("claim") or {}
    tail = (
        f"(n={entry.get('n_holdout')}, effect={entry.get('effect')}, "
        f"CI={entry.get('ci')}, p={entry.get('p_value')}, "
        f"claim \"{claim.get('text')}\" {claim.get('status')}, "
        f"verdict={'REJECT H0' if entry.get('reject') else 'not rejected'})"
    )
    design = entry.get("design")
    details = entry.get("details") or {}
    if design == "two_group" and outcome_kind == "binary":
        return (f"- {name}: flagged-group fail rate {entry.get('fail_rate_flagged')} "
                f"vs unflagged {entry.get('fail_rate_unflagged')} {tail}")
    if design == "two_group":
        return (f"- {name}: outcome mean {details.get('mean_flagged')} among flagged rows "
                f"vs {details.get('mean_unflagged')} among unflagged {tail}")
    if design == "stratified":
        return (f"- {name}: difference within strata, averaged over "
                f"{details.get('n_strata_used')} strata "
                f"(unadjusted {details.get('unadjusted_difference')}) {tail}")
    if design == "did":
        return (f"- {name}: effect {details.get('effect_where_modifier_1')} where the "
                f"modifier holds vs {details.get('effect_where_modifier_0')} where it "
                f"does not; difference of the two {tail}")
    return f"- {name}: outcome change per step up the fixed bins {tail}"


def holdout_confirm(
    report: "dict[str, Any]",
    holdout_rows: "list[dict[str, Any]]",
    *,
    outcome_col: str = "label",
    alpha: float = 0.05,
    judge: Any = None,
    judge_meta: "dict[str, Any] | None" = None,
    group_col: str | None = None,
    id_col: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    n_representatives: int = 3,
) -> "dict[str, Any]":
    """Re-test one exploratory report's recipes + hypotheses on held-out rows.

    *report* is the ``exploratory_report.json`` dict (``to_dict()`` of a live
    report works too). Returns the ``confirm_report.json`` payload; the caller
    persists it next to the exploratory report.

    *group_col* names the unit rows were cut from. Rows that share it are
    resampled together, so an interval reflects how many independent groups
    there are rather than how many rows. Without it rows are treated as
    independent, which is only right when they are.

    Every adjudicated signal makes one claim — the recipe's ``expect`` when it
    has one, otherwise "the effect is not zero" — and so does every numeric
    prediction a hypothesis states. All claims are corrected together with
    Benjamini-Hochberg at *alpha*; ``reject`` on a signal means its claim
    survived that correction. ``adjudication.method`` names the correction
    that was actually applied, and is ``"none"`` when no claim could be tested.
    """
    rows = [dict(r) for r in holdout_rows]
    outcome, outcome_kind, positive_label, outcome_note = outcome_values(rows, outcome_col)
    n_fail = int(sum(v for v in outcome if v is not None)) if outcome_kind == "binary" else 0
    clusters = [r.get(group_col) for r in rows] if group_col else None

    candidates_raw = [c for c in report.get("candidate_signals") or [] if isinstance(c, dict)]
    hypotheses = [h for h in report.get("hypotheses") or [] if isinstance(h, dict)]

    # ── 1. frozen-recipe re-evaluation on the held-out rows ─────────────────
    signal_verdicts: "list[dict[str, Any]]" = []
    estimates: "dict[int, DesignEstimate]" = {}
    recipes: "dict[int, SignalRecipe]" = {}
    for c in candidates_raw:
        entry: "dict[str, Any]" = {
            "name": c.get("name"),
            "display_name": c.get("display_name") or c.get("name"),
            "recipe": c.get("recipe"),
            "explore_effect": c.get("effect"),
            "explore_reject_in_sample": c.get("reject"),
        }
        signal_verdicts.append(entry)
        recipe_dict = c.get("recipe")
        if outcome_kind is None:
            entry.update(status="skipped", reason=outcome_note or "no usable outcome")
            continue
        if not isinstance(recipe_dict, dict) or recipe_dict.get("kind", "expr") != "expr":
            entry.update(status="skipped", reason="no expr recipe to re-evaluate")
            continue
        recipe = SignalRecipe.from_dict(recipe_dict)
        try:
            est = estimate_design(
                recipe, rows, outcome, clusters=clusters, alpha=alpha,
                n_boot=n_boot, seed=seed, reserved=(outcome_col,),
            )
        except (RecipeError, NotImplementedError) as exc:
            entry.update(status="skipped", reason=f"recipe error: {exc}")
            continue
        if not est.ok:
            entry.update(status="skipped", reason=est.reason, design=est.design)
            continue
        index = len(signal_verdicts) - 1
        estimates[index] = est
        recipes[index] = recipe
        entry.update(
            status="adjudicated",
            design=est.design,
            n_holdout=est.n,
            effect=est.effect,
            ci=est.ci,
            p_value=est.p_value,
            details=est.details,
        )
        if est.n_clusters is not None:
            entry["n_clusters"] = est.n_clusters
        if est.design == "two_group":
            entry["n_flagged"] = est.details["n_flagged"]
            if outcome_kind == "binary":
                entry["fail_rate_flagged"] = round(est.details["mean_flagged"], 4)
                entry["fail_rate_unflagged"] = round(est.details["mean_unflagged"], 4)

    # ── 2. numeric predictions carried by hypotheses, on the same estimates ──
    by_name: "dict[str, int]" = {}
    for index in estimates:
        by_name.setdefault(str(signal_verdicts[index].get("name")), index)
    prediction_results: "list[list[dict[str, Any]]]" = []
    for h in hypotheses:
        results: "list[dict[str, Any]]" = []
        for raw in h.get("predictions") or []:
            if not isinstance(raw, dict):
                continue
            item: "dict[str, Any]" = {
                "signal": raw.get("signal"), "relation": raw.get("relation"),
                "value": raw.get("value"),
            }
            parsed = parse_expectation(raw)
            target = by_name.get(str(raw.get("signal")))
            if parsed is None:
                item.update(status="not_testable", reason="malformed prediction")
            elif target is None:
                item.update(status="not_testable",
                            reason="no held-out estimate for that signal")
            else:
                relation, value = parsed
                est = estimates[target]
                item.update(
                    relation=relation, value=value, effect=est.effect, ci=est.ci,
                    p_value=bootstrap_p_value(est.draws, relation, value),
                    _p_contrary=contrary_p_value(est.draws, relation, value),
                )
            results.append(item)
        prediction_results.append(results)

    # ── 3. one correction family over every claim ───────────────────────────
    signal_claims = sorted(estimates)
    hypothesis_claims = [
        (hi, pi) for hi, results in enumerate(prediction_results)
        for pi, item in enumerate(results) if "p_value" in item
    ]
    p_values = [float(estimates[i].p_value or 1.0) for i in signal_claims] + [
        float(prediction_results[hi][pi]["p_value"]) for hi, pi in hypothesis_claims
    ]
    surviving = set(bh(p_values, alpha))

    for position, index in enumerate(signal_claims):
        est, entry = estimates[index], signal_verdicts[index]
        holds = position in surviving
        status = claim_status(holds, est.p_contrary, alpha)
        entry["claim"] = {
            "relation": est.relation, "value": est.value, "text": est.claim_text(),
            "status": status, "stated_by_recipe": recipes[index].expect is not None,
        }
        entry["raw_reject"] = bool((est.p_value or 1.0) <= alpha)
        entry["fdr_corrected"] = holds
        # "The effect is at most this large" holding is not a separation of
        # the outcome groups, so it never reads as a rejection.
        entry["reject"] = holds and est.relation != "abs<"
        if holds and n_representatives > 0:
            picked = representative_rows(
                est, rows, outcome, expr_columns(recipes[index].expr), k=n_representatives,
            )
            if picked:
                entry["representative_cases"] = {
                    side: [_row_ref(rows[i], i, id_col) for i in idxs]
                    for side, idxs in picked.items()
                }
    for offset, (hi, pi) in enumerate(hypothesis_claims):
        item = prediction_results[hi][pi]
        holds = (len(signal_claims) + offset) in surviving
        item["status"] = claim_status(holds, item.pop("_p_contrary"), alpha)

    n_claims = len(p_values)
    # Which tests the p-values in the family came from; each signal records
    # its own under details.p_value_method. A hypothesis prediction is always
    # read off the bootstrap replicates.
    p_value_methods = sorted(
        {str(estimates[i].details.get("p_value_method")) for i in signal_claims}
        | ({"percentile bootstrap"} if hypothesis_claims else set())
    )
    rejected = [str(e.get("name")) for e in signal_verdicts if e.get("reject")]
    meta: "dict[str, Any]" = {
        "method": "BH" if n_claims else "none",
        "alpha": alpha,
        "split": "held_out",
        "n_candidates": len(candidates_raw),
        "n_host_adjudicated": len(estimates),
        "n_in_family": n_claims,
        "n_signal_claims": len(signal_claims),
        "n_hypothesis_claims": len(hypothesis_claims),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "p_values": p_value_methods or None,
        "clustered_by": group_col,
    }
    if not n_claims:
        meta["note"] = "no claim could be re-tested on the held-out rows, so nothing was corrected"

    # ── 4. LLM judge per hypothesis, grounded in the held-out table ─────────
    evidence_lines = [
        _evidence_line(e, outcome_kind)
        for e in signal_verdicts
        if e.get("status") == "adjudicated"
    ] or ["- (no recipe could be re-evaluated on the held-out split)"]
    evidence = "\n".join(evidence_lines)

    hypothesis_verdicts: "list[dict[str, Any]]" = []
    for h, results in zip(hypotheses, prediction_results):
        verdict = dict(h)
        host_verdict = _host_verdict(results)
        if results:
            verdict["prediction_results"] = results
        verdict["host_verdict"] = host_verdict
        if judge is None:
            verdict.update(verdict="not_judged", reasoning="", needs_surgery=True)
            hypothesis_verdicts.append(verdict)
            continue
        checked = ""
        if results:
            checked = "\n\nPREDICTIONS THE HOST ALREADY CHECKED for this hypothesis:\n" + "\n".join(
                f"- {r.get('signal')} {r.get('relation')} {r.get('value')}: "
                f"{r.get('status')} (effect={r.get('effect')}, CI={r.get('ci')})"
                for r in results
            )
        prompt = JUDGE_PROMPT.format(
            statement=h.get("statement", ""),
            basis=h.get("basis", ""),
            test_design=h.get("test_design", ""),
            n_rows=len(rows),
            evidence=evidence + checked,
        )
        try:
            graded = ask_judge(judge, prompt, _HoldoutVerdict, role="holdout_judge").output
        except JudgeError as exc:  # a judge outage must not sink the signal verdicts
            # "not_testable" would be a claim about the hypothesis; the judge made none.
            verdict.update(verdict="not_judged", reasoning="", needs_surgery=True,
                           judge_error=str(exc))
        else:
            verdict.update(verdict=graded.verdict, reasoning=graded.reasoning,
                           needs_surgery=graded.needs_surgery)
            if (not verdict.get("experiment") and graded.experiment is not None
                    and not graded.experiment.is_empty()):
                verdict["experiment"] = graded.experiment.model_dump()
        hypothesis_verdicts.append(verdict)

    return {
        "phase": "holdout_confirm",
        "split": "held_out",
        "n_validate_rows": len(rows),
        "n_validate_fail": n_fail,
        "alpha": alpha,
        "outcome": {
            "column": outcome_col,
            "kind": outcome_kind,
            "positive_label": positive_label,
            "note": outcome_note,
        },
        "adjudication": meta,
        "signal_verdicts": signal_verdicts,
        "hypothesis_verdicts": hypothesis_verdicts,
        "judge": judge_meta if judge is not None else None,
    }


def _host_verdict(results: "list[dict[str, Any]]") -> str:
    """A hypothesis's standing on its own numeric predictions.

    ``supported`` when every prediction holds, ``refuted`` when any is
    contradicted, ``inconclusive`` when some stay open, and ``none`` when it
    stated no prediction the held-out rows could test.
    """
    statuses = [r.get("status") for r in results if r.get("status") != "not_testable"]
    if not statuses:
        return "none"
    if "contradicted" in statuses:
        return "refuted"
    if all(s == "holds" for s in statuses):
        return "supported"
    return "inconclusive"
