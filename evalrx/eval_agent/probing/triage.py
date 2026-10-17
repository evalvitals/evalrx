"""Route labels from the triage observations, each traced to case, field and action.

Every check on every case ends in one of four states — ``fired``, ``clear``,
``untested`` (with the reason) or ``not_applicable`` (the field has no meaning
on this task) — so a missing measurement is never read as a negative one.

* **H** (output or grading suspect): the answer-extraction and termination
  audits. Fields defined through the gold answer or the FAIL label
  (``extraction_suspect``, ``extraction_point_miss``) route cases to an audit
  but are excluded from the FAIL-vs-PASS routing signal, which they would
  produce by construction. Truncation fields are untested when the host, not
  the backend, supplied ``finish_reason``.
* **N** (insensitive to a media swap): only a real swap with a different-content
  donor and a non-empty baseline answer counts; a swap that fell back to drop
  is a different intervention and leaves N untested.
* **U** is untested in MVP-0 (no repeated sampling or perturbation runs), and
  **S** is not assigned: it needs the U and other behavioural checks first.
* **X**: no H or N signal fired; the case stays unexplained, with the reasons.

``attention_knockout`` is not a route: its per-case margins and fidelity
verdict are attached to each case as the first N-layer measurement.
"""

from __future__ import annotations

from typing import Any

from evalrx.eval_agent.probing.actions import OK, Observation

FIRED, CLEAR, UNTESTED, NOT_APPLICABLE = "fired", "clear", "untested", "not_applicable"

#: (analyzer, field, label link, kind). A label link names what the field's
#: definition already reads besides the model output.
H_FIELDS: tuple[tuple[str, str, "str | None", str], ...] = (
    ("answer_extraction_audit", "extraction_suspect", "FAIL label + gold", "flag"),
    ("answer_extraction_audit", "extraction_point_miss", "gold", "flag"),
    ("answer_extraction_audit", "output_truncated", None, "truncation"),
    ("answer_extraction_audit", "gave_up", None, "flag"),
    ("termination_audit", "looks_truncated", None, "truncation"),
    ("termination_audit", "termination_class", None, "class"),
    ("termination_audit", "gave_up", None, "flag"),
)
_FIRING_CLASSES = {"truncated", "degenerate", "gave_up"}
_KNOCKOUT_FIELDS = ("ko_margin_clean", "ko_delta_full", "ko_rel_drop_full", "ko_flips_top1",
                    "ko_delta_early", "ko_delta_mid", "ko_delta_late")


def finish_reason_source(case: Any, host_assumed: bool) -> str:
    """``host_assumed`` | ``backend`` | ``heuristic`` — where truncation evidence comes from."""
    if host_assumed:
        return "host_assumed"
    meta = getattr(case, "metadata", None) or {}
    return "backend" if meta.get("finish_reason") in ("stop", "length") else "heuristic"


def _unavailable(obs: "Observation | None", analyzer: str) -> "str | None":
    if obs is None:
        return f"{analyzer} not in the plan or not executed"
    if obs.status != OK:
        return f"{analyzer} {obs.status}: {obs.error or ''}".strip()
    return None


def _h_field(obs: "Observation | None", analyzer: str, field: str, link: "str | None", kind: str,
             cid: str, provenance: str, answer_contract: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"analyzer": analyzer, "field": field, "label_link": link}
    why = _unavailable(obs, analyzer)
    if why is not None:
        return {**out, "state": UNTESTED, "reason": why}
    out["action_id"] = obs.action_id
    row = obs.per_case().get(cid)
    if row is None:
        return {**out, "state": UNTESTED, "reason": obs.missing.get(cid, "no per_case row")}
    if field not in row:
        return {**out, "state": UNTESTED, "reason": str(row.get("skipped") or f"{field} not reported")}
    value = row[field]
    out["value"] = value
    if kind == "truncation":
        out["provenance"] = provenance
        if provenance == "host_assumed":
            return {**out, "state": UNTESTED,
                    "reason": "finish_reason='stop' was set by the host for a short-answer task; "
                              "truncation was not measured"}
        return {**out, "state": FIRED if value else CLEAR}
    if kind == "class":
        if value == "no_answer" and not answer_contract:
            return {**out, "state": NOT_APPLICABLE,
                    "reason": "the task declares no answer tag or output contract, so 'no_answer' "
                              "only means none was found"}
        return {**out, "state": FIRED if value in _FIRING_CLASSES
                or (value == "no_answer" and answer_contract) else CLEAR}
    return {**out, "state": FIRED if value else CLEAR}


def _combine(fields: list[dict[str, Any]]) -> str:
    states = {f["state"] for f in fields}
    if FIRED in states:
        return FIRED
    if CLEAR in states:
        return CLEAR
    return UNTESTED


def _n_check(obs: "Observation | None", cid: str) -> dict[str, Any]:
    why = _unavailable(obs, "modality_ablation")
    if why is not None:
        return {"state": UNTESTED, "reason": why}
    out: dict[str, Any] = {"action_id": obs.action_id, "mode": (obs.params or {}).get("mode")}
    if out["mode"] != "swap":
        return {**out, "state": UNTESTED, "reason": f"modality_ablation ran mode={out['mode']!r}, not swap"}
    row = obs.per_case().get(cid)
    if row is None:
        return {**out, "state": UNTESTED, "reason": obs.missing.get(cid, "no media slot to swap")}
    out["base_answer"] = row.get("ablation_base_answer")
    if not str(out["base_answer"] or "").strip():
        return {**out, "state": UNTESTED, "reason": "empty baseline answer; the comparison is not reliable"}
    slots = []
    for key in sorted(row):
        if not key.startswith("grounded_in_"):
            continue
        slot = key[len("grounded_in_"):]
        entry = {"slot": slot, "donor": row.get(f"ablation_{slot}_donor"),
                 "swapped_answer": row.get(f"ablation_{slot}_answer")}
        if row.get(f"ablation_{slot}_is_swap") is not True:
            entry.update(state=UNTESTED, reason="no case held different media; the slot was dropped, "
                                                "which is a different intervention")
        else:
            entry.update(state=FIRED if row[key] is False else CLEAR, answer_changed=bool(row[key]))
        slots.append(entry)
    if not slots:
        return {**out, "state": UNTESTED, "reason": "no slot was ablated"}
    state = _combine(slots)
    if state == UNTESTED:
        out["reason"] = "; ".join(f"{s['slot']}: {s['reason']}" for s in slots)
    return {**out, "state": state, "slots": slots}


def _knockout(obs: "Observation | None", cid: str) -> dict[str, Any]:
    if obs is None:
        return {"state": UNTESTED, "reason": "attention_knockout not in the plan or not executed"}
    out: dict[str, Any] = {"action_id": obs.action_id, "status": obs.status,
                           "fidelity": (obs.fidelity or {}).get("status")}
    if obs.status != OK:
        return {**out, "state": UNTESTED, "reason": obs.error or obs.status}
    row = obs.per_case().get(cid)
    if row is None:
        return {**out, "state": UNTESTED, "reason": obs.missing.get(cid, "no per_case row")}
    return {**out, "state": "measured", **{k: row[k] for k in _KNOCKOUT_FIELDS if k in row}}


def triage_cases(cases: Any, observations: dict[str, Observation], *, host_assumed_finish: bool,
                 answer_contract: bool) -> list[dict[str, Any]]:
    """One record per case: label, stratum, every H/N check, knockout row, routes."""
    from evalrx.eval_agent.probing.sampler import gold_stratum

    records = []
    for case in cases:
        cid = str(case.id)
        provenance = finish_reason_source(case, host_assumed_finish)
        h_fields = [
            _h_field(observations.get(a), a, f, link, kind, cid, provenance, answer_contract)
            for a, f, link, kind in H_FIELDS
        ]
        free = [f for f in h_fields if f["label_link"] is None]
        disagreements = []
        trunc = {f["analyzer"]: f for f in h_fields if f.get("provenance") and f["state"] in (FIRED, CLEAR)}
        if len(trunc) == 2 and len({f["state"] for f in trunc.values()}) == 2:
            disagreements.append({"fields": ["answer_extraction_audit.output_truncated",
                                             "termination_audit.looks_truncated"],
                                  "note": "the two truncation reads disagree; audit before relying on either"})
        h = {
            "state": _combine(h_fields),
            "state_label_free": _combine(free),
            "fired": [f"{f['analyzer']}.{f['field']}" for f in h_fields if f["state"] == FIRED],
            "untested": {f"{f['analyzer']}.{f['field']}": f.get("reason") for f in h_fields
                         if f["state"] == UNTESTED},
            "fields": h_fields,
            "disagreements": disagreements,
        }
        n = _n_check(observations.get("modality_ablation"), cid)
        routes = [tag for tag, state in (("H", h["state"]), ("N", n["state"])) if state == FIRED]
        x_reasons = []
        if not routes:
            if h["state"] == UNTESTED:
                x_reasons.append("H untested")
            if n["state"] == UNTESTED:
                x_reasons.append(f"N untested: {n.get('reason')}")
            x_reasons.append("U untested: MVP-0 runs no repeated sampling or meaning-preserving "
                             "perturbation; S needs those checks first")
            routes = ["X"]
        records.append({
            "case_id": cid,
            "label": getattr(getattr(case, "label", None), "value", None),
            "stratum": gold_stratum(case),
            "routes": routes,
            "x_reasons": x_reasons,
            "H": h,
            "N": n,
            "U": {"state": UNTESTED, "reason": "not run in MVP-0"},
            "knockout": _knockout(observations.get("attention_knockout"), cid),
        })
    return records


def _rate(records: list[dict[str, Any]], label: str, state_of: Any) -> dict[str, Any]:
    group = [r for r in records if r["label"] == label]
    tested = [r for r in group if state_of(r) in (FIRED, CLEAR)]
    fired = [r for r in tested if state_of(r) == FIRED]
    return {"n": len(group), "n_tested": len(tested), "n_fired": len(fired),
            "rate": round(len(fired) / len(tested), 4) if tested else None}


def _mean(values: list[float]) -> "float | None":
    return round(sum(values) / len(values), 4) if values else None


def _n_by_base_answer(records: list[dict[str, Any]], max_answers: int = 5) -> "dict[str, Any] | None":
    """N split by the unablated answer, for categorical answers only.

    An unchanged answer is only informative when the donor media should have
    changed it. Whether it should is not known here — on POPE, a random donor
    image usually lacks the asked-about object too, so an unchanged "No" is
    the expected answer while an unchanged "Yes" is not. The split shows which
    answers the N signal comes from instead of guessing a donor's gold.
    """
    from evalrx.analyzers.perturbation.prompt_contrast import _norm_answer

    tested = [r for r in records if r["N"]["state"] in (FIRED, CLEAR)]
    answers = sorted({_norm_answer(str(r["N"].get("base_answer") or "")) for r in tested})
    if not answers or len(answers) > max_answers:
        return None
    out: dict[str, Any] = {}
    for answer in answers:
        group = [r for r in tested if _norm_answer(str(r["N"].get("base_answer") or "")) == answer]
        out[answer] = {label: {"n_tested": sum(1 for r in group if r["label"] == label),
                               "n_fired": sum(1 for r in group if r["label"] == label and r["N"]["state"] == FIRED)}
                       for label in ("fail", "pass")}
    out["_note"] = ("unchanged-after-swap counts by the unablated answer; an unchanged answer that the "
                    "donor media would also warrant is not evidence of ignoring the media")
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-route FAIL/PASS rates on this sample, and the descriptive knockout summary.

    ``routing_signal = max(0, P(tag|FAIL) - P(tag|PASS))`` steers where the
    next actions go on THIS sample; it is not an explained share of failures
    and says nothing about causes. H uses only label-free fields here.
    """
    tags = {
        "H_label_free": lambda r: r["H"]["state_label_free"],
        "N": lambda r: r["N"]["state"],
    }
    out: dict[str, Any] = {"n_cases": len(records)}
    for tag, state_of in tags.items():
        fail, pas = _rate(records, "fail", state_of), _rate(records, "pass", state_of)
        signal = (round(max(0.0, fail["rate"] - pas["rate"]), 4)
                  if fail["rate"] is not None and pas["rate"] is not None else None)
        out[tag] = {"fail": fail, "pass": pas, "routing_signal": signal}
    out["H_any_audit_only"] = {
        label: _rate(records, label, lambda r: r["H"]["state"]) for label in ("fail", "pass")
    }
    out["N_by_base_answer"] = _n_by_base_answer(records)
    out["X"] = {label: sum(1 for r in records if r["label"] == label and r["routes"] == ["X"])
                for label in ("fail", "pass")}
    ko: dict[str, Any] = {}
    for label in ("fail", "pass"):
        rows = [r["knockout"] for r in records if r["label"] == label and r["knockout"]["state"] == "measured"]
        ko[label] = {"n_measured": len(rows),
                     "mean_ko_delta_full": _mean([float(r["ko_delta_full"]) for r in rows if "ko_delta_full" in r]),
                     "mean_ko_delta_late": _mean([float(r["ko_delta_late"]) for r in rows if "ko_delta_late" in r])}
    ko["fidelity"] = sorted({str(r["knockout"].get("fidelity")) for r in records})
    ko["_note"] = "descriptive only: margin deltas of the gold first token, no test, no mechanism claim"
    out["attention_knockout"] = ko
    return out
