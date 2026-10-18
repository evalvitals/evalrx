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
* **N** (an answer that should change with the media does not): only a real
  swap with a non-empty baseline answer counts, and on few-answer tasks only a
  counterfactual donor (the same question's gold on that media differs from the model's answer);
  a content-only swap there, or a swap that fell back to drop, leaves N untested.
* **U** (an answer that moves): two parts, both required. ``self_consistency``
  samples at the host's sampling settings against the Stage-0 answer (greedy
  settings leave it untested: identical samples are not a measurement), and
  ``prompt_contrast`` wrappers that keep the question verbatim against the
  re-asked baseline (a wrapper whose answer is unreadable is an invalid
  control, not an unstable answer). Answers are compared by the task's answer
  key, never graded. Any readable answer that differs fires U.
* **S** (a stable failure): a FAIL case on which every required check ran and
  none fired — H, N (cases with media) and both U parts. S is an
  investigation route, not a cause; the S readings attached to every case
  (the threshold instruction, the question word on the media tokens, whether
  the gold answer was readable before the output) are descriptive.
* **X**: no route fired and a required check is missing, with the reasons. A
  missing check never makes a case S. A PASS case with every check clear has
  no route (it is a stable control).

``attention_knockout`` is not a route: its per-case margins and fidelity
verdict are attached to each case as the first N-layer measurement.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

from evalrx.analyzers.perturbation.prompt_contrast import _norm_answer
from evalrx.core.case import MEDIA_SLOTS
from evalrx.eval_agent.probing.actions import OK, Observation
from evalrx.eval_agent.probing.evidence import StepEvidence, pick

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
_EMBED_FIELDS = ("el_query_word", "el_query_sim", "el_frac_specific")
_PATCHSCOPE_FIELDS = ("ps_final_prob", "ps_peak_prob", "ps_peak_depth_frac", "ps_late_drop",
                      "ps_first_depth_frac", "ps_media_query_hits", "ps_media_query_found")

AnswerFn = Callable[[Any, str], "str | None"]

#: The plan steps the H/N/U checks (and so the routes, S and X) are read from.
#: Every other step only adds descriptive readings to a case.
ROUTE_STEPS = frozenset({"answer_extraction_audit", "termination_audit", "modality_ablation",
                         "self_consistency", "prompt_reword"})


def default_answer(case: Any, text: str) -> str:
    """Coarse answer key when the host gives none: yes/no, else a normalised prefix."""
    return _norm_answer(str(text or ""))


def _has_media(case: Any) -> bool:
    inputs = getattr(case, "inputs", None)
    return any(getattr(inputs, slot, None) is not None for slot in MEDIA_SLOTS)


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


def _n_check(obs: "Observation | None", case: Any, needs_counterfactual: bool) -> dict[str, Any]:
    cid = str(case.id)
    if not _has_media(case):
        return {"state": NOT_APPLICABLE, "reason": "the case has no media to swap"}
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
            slots.append(entry)
            continue
        kind = row.get(f"ablation_{slot}_donor_kind") or "content"
        expected = row.get(f"ablation_{slot}_donor_expected")
        entry.update(donor_kind=kind, donor_expected=expected, answer_changed=bool(row[key]))
        if kind == "counterfactual":
            # the same question's gold on the donor media differs from the model's answer:
            # a correct answer there has to be different
            entry.update(state=FIRED if row[key] is False else CLEAR,
                         follows_donor=_norm_answer(str(entry["swapped_answer"] or "")) == expected)
        elif needs_counterfactual:
            entry.update(state=UNTESTED,
                         reason="no counterfactual donor in the partition (same question, different "
                                "media, gold different from the model's answer); on a few-answer task "
                                "an unchanged answer after a content-only swap may simply be correct")
        else:
            entry.update(state=FIRED if row[key] is False else CLEAR,
                         assumption="open-answer task: different media is assumed to change the answer")
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


def _samples_vary(gen: dict[str, Any]) -> bool:
    try:
        temperature = float(gen.get("temperature") or 0.0)
    except (TypeError, ValueError):
        temperature = 0.0
    return bool(gen.get("do_sample")) or temperature > 0.0


def _u_sampling(obs: "Observation | None", case: Any, answer_fn: AnswerFn) -> dict[str, Any]:
    why = _unavailable(obs, "self_consistency")
    if why is not None:
        return {"part": "sampling", "state": UNTESTED, "reason": why}
    cid = str(case.id)
    gen = (obs.params or {}).get("gen_kwargs") or {}
    out: dict[str, Any] = {"part": "sampling", "action_id": obs.action_id, "gen_kwargs": gen}
    if not _samples_vary(gen):
        return {**out, "state": UNTESTED, "reason": "greedy decoding (no temperature or do_sample in "
                "gen_kwargs): repeated samples are identical by construction"}
    samples = ((obs.per_case().get(cid) or {}).get("artifacts") or {}).get("samples")
    if not samples:
        return {**out, "state": UNTESTED, "reason": obs.missing.get(cid, "no samples kept for this case")}
    base = answer_fn(case, str(getattr(case, "observed", "") or ""))
    if base is None:
        return {**out, "state": UNTESTED, "reason": "the task has no categorical answer to compare"}
    if not base:
        return {**out, "state": UNTESTED, "reason": "the Stage-0 output commits to no readable answer"}
    answers = [answer_fn(case, str(s)) for s in samples]
    readable = [a for a in answers if a]
    out.update(base_answer=base, n_samples=len(answers), n_unreadable=len(answers) - len(readable),
               answers=dict(Counter(readable)))
    if not readable:
        return {**out, "state": UNTESTED, "reason": "no sample commits to a readable answer"}
    out["agreement"] = round(sum(a == base for a in readable) / len(readable), 4)
    return {**out, "state": FIRED if out["agreement"] < 1.0 else CLEAR}


def _outputs(obs: Observation) -> dict[str, dict[str, str]]:
    return ((obs.findings or {}).get("artifacts") or {}).get("outputs") or {}


def _u_reword(obs: "Observation | None", case: Any, answer_fn: AnswerFn) -> dict[str, Any]:
    why = _unavailable(obs, "prompt_contrast (rewording)")
    if why is not None:
        return {"part": "reword", "state": UNTESTED, "reason": why}
    cid = str(case.id)
    outputs = _outputs(obs)
    out: dict[str, Any] = {"part": "reword", "action_id": obs.action_id}
    if cid not in (outputs.get("baseline") or {}):
        return {**out, "state": UNTESTED, "reason": obs.missing.get(cid, "case not re-asked")}
    base = answer_fn(case, outputs["baseline"][cid])
    if base is None:
        return {**out, "state": UNTESTED, "reason": "the task has no categorical answer to compare"}
    if not base:
        return {**out, "state": UNTESTED, "reason": "the re-asked question commits to no readable answer"}
    variants = {s: answer_fn(case, texts[cid]) for s, texts in outputs.items() if s != "baseline" and cid in texts}
    valid = {s: a for s, a in variants.items() if a}
    out.update(base_answer=base, answers=valid, invalid_controls=sorted(s for s in variants if s not in valid),
               baseline_matches_stage0=answer_fn(case, str(getattr(case, "observed", "") or "")) == base)
    if not valid:
        return {**out, "state": UNTESTED,
                "reason": f"no wrapper gave a readable answer (invalid controls: {out['invalid_controls']})"}
    out["changed"] = sorted(s for s, a in valid.items() if a != base)
    return {**out, "state": FIRED if out["changed"] else CLEAR}


def _u_check(sampling: "Observation | None", reword: "Observation | None", case: Any,
             answer_fn: AnswerFn) -> dict[str, Any]:
    """U fires when either part fires; it is clear only when both parts ran and are clear."""
    parts = [_u_sampling(sampling, case, answer_fn), _u_reword(reword, case, answer_fn)]
    states = [p["state"] for p in parts]
    state = FIRED if FIRED in states else CLEAR if all(st == CLEAR for st in states) else UNTESTED
    out: dict[str, Any] = {"state": state, "parts": parts}
    if state == UNTESTED:
        out["reason"] = "; ".join(f"{p['part']}: {p['reason']}" for p in parts if p["state"] == UNTESTED)
    return out


def _s_threshold(obs: "Observation | None", case: Any, answer_fn: AnswerFn) -> dict[str, Any]:
    """The answer under the changed decision criterion, against the re-asked baseline."""
    if obs is None:
        return {"state": UNTESTED, "reason": "threshold instruction not in the plan for this task"}
    why = _unavailable(obs, "prompt_contrast (threshold)")
    if why is not None:
        return {"state": UNTESTED, "reason": why}
    cid = str(case.id)
    outputs, scored = _outputs(obs), (obs.findings or {}).get("by_strategy") or {}
    variant = next((s for s in outputs if s != "baseline"), None)
    if variant is None or cid not in (outputs.get("baseline") or {}) or cid not in (outputs.get(variant) or {}):
        return {"state": UNTESTED, "reason": obs.missing.get(cid, "case not re-asked")}
    base, alt = answer_fn(case, outputs["baseline"][cid]), answer_fn(case, outputs[variant][cid])
    out: dict[str, Any] = {"action_id": obs.action_id, "strategy": variant, "base_answer": base, "answer": alt}
    if not base or not alt:
        return {**out, "state": UNTESTED, "reason": "an answer was unreadable or the task has no answer key"}
    before, after = (scored.get("baseline") or {}).get(cid), (scored.get(variant) or {}).get(cid)
    out.update(changed=alt != base, fixed=before == 0.0 and after == 1.0, broken=before == 1.0 and after == 0.0)
    return {**out, "state": "measured"}


def _measured_row(obs: "Observation | None", cid: str, analyzer: str, keys: tuple[str, ...]) -> dict[str, Any]:
    why = _unavailable(obs, analyzer)
    if why is not None:
        return {"state": UNTESTED, "reason": why}
    row = obs.per_case().get(cid)
    if row is None:
        return {"state": UNTESTED, "action_id": obs.action_id, "reason": obs.missing.get(cid, "no per_case row")}
    return {"state": "measured", "action_id": obs.action_id, **{k: row[k] for k in keys if k in row}}


def _s_patchscope(obs: "Observation | None", cid: str) -> dict[str, Any]:
    out = _measured_row(obs, cid, "patchscope", _PATCHSCOPE_FIELDS)
    if out["state"] == "measured":
        out["target"] = (obs.params or {}).get("target")
        out["fidelity"] = (obs.fidelity or {}).get("status")
        # ranked first at some read layer before the last: formed, then not output
        out["target_first_before_last"] = float(out.get("ps_first_depth_frac", 1.0)) < 1.0
    return out


def triage_cases(cases: Any, observations: "dict[str, Observation | StepEvidence]", *,
                 host_assumed_finish: bool,
                 answer_contract: bool, n_needs_counterfactual: bool = False,
                 answer_fn: "AnswerFn | None" = None) -> list[dict[str, Any]]:
    """One record per case: label, stratum, every H/N/U check, routes, knockout and S readings.

    *observations* are keyed by plan step (``prompt_reword`` and
    ``prompt_threshold`` are the two ``prompt_contrast`` steps): one
    observation, or a :class:`StepEvidence` whose chunks each measured part of
    the sample (each case is read from the chunk that measured it).

    *n_needs_counterfactual* (few-answer tasks: yes/no, letters): a swap only
    tests N when its donor is a counterfactual — the same question's gold on that
    media differs from the model's answer — because a content-only swap there often leaves the
    correct answer unchanged.

    *answer_fn* ``(case, output) -> key`` reads the answer an output commits to
    (``""``: none; ``None``: the task has no categorical answer); default
    :func:`default_answer`.
    """
    from evalrx.eval_agent.probing.sampler import gold_stratum

    answer = answer_fn or default_answer
    records = []
    for case in cases:
        cid = str(case.id)
        provenance = finish_reason_source(case, host_assumed_finish)
        h_fields = [
            _h_field(pick(observations, a, cid), a, f, link, kind, cid, provenance, answer_contract)
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
        n = _n_check(pick(observations, "modality_ablation", cid), case, n_needs_counterfactual)
        u = _u_check(pick(observations, "self_consistency", cid), pick(observations, "prompt_reword", cid),
                     case, answer)
        label = getattr(getattr(case, "label", None), "value", None)
        checks = {"H": h["state"], "N": n["state"], "U": u["state"]}
        routes = [tag for tag, state in checks.items() if state == FIRED]
        missing = [tag for tag, state in checks.items() if state == UNTESTED]
        # every required check ran and none fired: a stable failure, or a stable control
        stable = not routes and not missing
        x_reasons = []
        if not routes:
            if stable:
                routes = ["S"] if label == "fail" else []
            else:
                why = {"H": None, "N": n.get("reason"), "U": u.get("reason")}
                x_reasons = [f"{tag} untested" + (f": {why[tag]}" if why[tag] else "") for tag in missing]
                routes = ["X"]
        records.append({
            "case_id": cid,
            "label": label,
            "stratum": gold_stratum(case),
            "routes": routes,
            "stable": stable,
            "x_reasons": x_reasons,
            "H": h,
            "N": n,
            "U": u,
            "knockout": _knockout(pick(observations, "attention_knockout", cid), cid),
            "S_readings": {
                "threshold": _s_threshold(pick(observations, "prompt_threshold", cid), case, answer),
                "embed_lens": _measured_row(pick(observations, "embed_lens", cid), cid, "embed_lens",
                                           _EMBED_FIELDS),
                "patchscope": _s_patchscope(pick(observations, "patchscope", cid), cid),
            },
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

    Descriptive: shows which answers the tested N checks come from. Without
    counterfactual donors it is also the only guard against reading an
    unchanged answer the donor media would warrant anyway (on POPE, a random
    image usually lacks the asked-about object, so "No" stays correct).
    """
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


def _n_donors(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Which donor each case's swap got, and how often a counterfactual swap
    moved the answer to the donor's gold (the model followed the new media)."""
    out: dict[str, Any] = {}
    for label in ("fail", "pass"):
        group = [r for r in records if r["label"] == label]
        kinds = [(r["N"].get("slots") or [{}])[0] for r in group]
        cf = [s for s in kinds if s.get("donor_kind") == "counterfactual"]
        out[label] = {
            "n": len(group),
            "counterfactual": len(cf),
            "content": sum(1 for s in kinds if s.get("donor_kind") == "content"),
            "no_swap": sum(1 for s in kinds if not s.get("donor_kind")),
            "counterfactual_followed_donor": sum(1 for s in cf if s.get("follows_donor")),
        }
    return out


def _threshold_table(records: list[dict[str, Any]]) -> "dict[str, Any] | None":
    """Threshold-instruction outcomes by label and stratum (every measured case)."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        t = r["S_readings"]["threshold"]
        if t["state"] != "measured":
            continue
        g = groups.setdefault((str(r["label"]), str(r["stratum"])), {
            "label": r["label"], "stratum": r["stratum"], "n_measured": 0, "n_changed": 0,
            "n_fixed": 0, "n_broken": 0})
        g["n_measured"] += 1
        g["n_changed"] += int(bool(t["changed"]))
        g["n_fixed"] += int(bool(t["fixed"]))
        g["n_broken"] += int(bool(t["broken"]))
    if not groups:
        return None
    return {"groups": [groups[k] for k in sorted(groups)],
            "_note": ("the same cases re-asked with the instruction against re-asked unchanged; fixed = "
                      "wrong to right, broken = right to wrong. Fixing missed objects while breaking correct "
                      "negatives moves the decision criterion; it does not by itself explain the error")}


#: Who the S readings are summarised over: the S route, the other failures, and
#: the PASS cases on which every check ran clear (the controls).
_COHORTS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "S": lambda r: r["label"] == "fail" and r["stable"],
    "fail_not_stable": lambda r: r["label"] == "fail" and not r["stable"],
    "stable_pass": lambda r: r["label"] == "pass" and r["stable"],
}


def _s_readings(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Embed-lens and Patchscope readings per cohort and stratum; descriptive only."""
    out: dict[str, Any] = {}
    for name, member in _COHORTS.items():
        for stratum in sorted({r["stratum"] for r in records if member(r)}):
            group = [r for r in records if member(r) and r["stratum"] == stratum]
            el = [r["S_readings"]["embed_lens"] for r in group
                  if r["S_readings"]["embed_lens"]["state"] == "measured"]
            ps = [r["S_readings"]["patchscope"] for r in group
                  if r["S_readings"]["patchscope"]["state"] == "measured"]
            read = [x for x in ps if "ps_media_query_hits" in x]
            out.setdefault(name, {})[stratum] = {
                "n": len(group),
                "embed_lens": {"n": len(el), "mean_query_sim": _mean(
                    [float(x["el_query_sim"]) for x in el if "el_query_sim" in x])},
                "patchscope": {
                    "n": len(ps),
                    "n_target_first_before_last": sum(1 for x in ps if x["target_first_before_last"]),
                    "mean_peak_prob": _mean([float(x["ps_peak_prob"]) for x in ps if "ps_peak_prob" in x]),
                    "mean_final_prob": _mean([float(x["ps_final_prob"]) for x in ps if "ps_final_prob" in x]),
                    "mean_late_drop": _mean([float(x["ps_late_drop"]) for x in ps if "ps_late_drop" in x]),
                    "n_media_query_read": len(read),
                    "n_media_query_hit": sum(1 for x in read if x["ps_media_query_hits"] > 0),
                },
            }
    out["_note"] = ("descriptive only, no test: el_query_sim = the case's own question word against the media "
                    "tokens; target_first_before_last = the patchscope target (gold) ranked first at some layer "
                    "before the last; media_query_hit = the question word decoded off some media token")
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def difference_interval(k1: int, n1: int, k2: int, n2: int) -> "list[float] | None":
    """95% interval of ``k1/n1 - k2/n2`` (Newcombe's hybrid score method), ``None`` if a side is empty."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = _wilson(k1, n1), _wilson(k2, n2)
    d = p1 - p2
    return [round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
            round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4)]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-route FAIL/PASS rates on this sample, the S cohort and its descriptive readings.

    ``routing_signal = max(0, P(tag|FAIL) - P(tag|PASS))`` steers where the
    next actions go on THIS sample; it is not an explained share of failures
    and says nothing about causes. ``signal_ci95`` is the 95% interval of the
    unclipped difference: how sure this sample is that the difference is not
    zero, which the expansion rule reads. H uses only label-free fields here.
    """
    tags = {
        "H_label_free": lambda r: r["H"]["state_label_free"],
        "N": lambda r: r["N"]["state"],
        "U": lambda r: r["U"]["state"],
    }
    out: dict[str, Any] = {"n_cases": len(records)}
    for tag, state_of in tags.items():
        fail, pas = _rate(records, "fail", state_of), _rate(records, "pass", state_of)
        signal = (round(max(0.0, fail["rate"] - pas["rate"]), 4)
                  if fail["rate"] is not None and pas["rate"] is not None else None)
        out[tag] = {"fail": fail, "pass": pas, "routing_signal": signal,
                    "signal_ci95": difference_interval(fail["n_fired"], fail["n_tested"],
                                                       pas["n_fired"], pas["n_tested"])}
    out["H_any_audit_only"] = {
        label: _rate(records, label, lambda r: r["H"]["state"]) for label in ("fail", "pass")
    }
    out["N_by_base_answer"] = _n_by_base_answer(records)
    out["N_donors"] = _n_donors(records)
    out["U_parts"] = {
        part: {label: _rate(records, label, lambda r, i=i: r["U"]["parts"][i]["state"]) for label in ("fail", "pass")}
        for i, part in enumerate(("sampling", "reword"))
    }
    out["S"] = {"fail": sum(1 for r in records if r["routes"] == ["S"]),
                "stable_pass": sum(1 for r in records if r["label"] == "pass" and r["stable"])}
    out["X"] = {label: sum(1 for r in records if r["label"] == label and r["routes"] == ["X"])
                for label in ("fail", "pass")}
    out["S_threshold"] = _threshold_table(records)
    out["S_readings"] = _s_readings(records)
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


# -- follow-up readings ------------------------------------------------------------


def _numeric(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if key == "sample_id" or key.startswith("_"):
            continue
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def _contrast_values(obs: Observation, case: Any, answer_fn: AnswerFn) -> dict[str, float]:
    """``prompt_contrast``: per strategy, whether the answer moved off the re-asked
    question (by answer key) and whether that fixed or broke it (by the task's scorer)."""
    cid = str(case.id)
    outputs, scored = _outputs(obs), (obs.findings or {}).get("by_strategy") or {}
    if cid not in (outputs.get("baseline") or {}):
        return {}
    base = answer_fn(case, outputs["baseline"][cid])
    before = (scored.get("baseline") or {}).get(cid)
    out: dict[str, float] = {}
    for strategy, texts in outputs.items():
        if strategy == "baseline" or cid not in texts:
            continue
        alt = answer_fn(case, texts[cid])
        if base and alt:
            out[f"changed_{strategy}"] = float(alt != base)
        after = (scored.get(strategy) or {}).get(cid)
        if before is not None and after is not None:
            out[f"fixed_{strategy}"] = float(before == 0.0 and after == 1.0)
            out[f"broken_{strategy}"] = float(before == 1.0 and after == 0.0)
    return out


def follow_up_values(evidence: StepEvidence, case: Any, answer_fn: "AnswerFn | None" = None) -> dict[str, Any]:
    """One case's numeric readings from a follow-up step, or why there are none."""
    cid = str(case.id)
    obs = evidence.for_case(cid)
    if obs is None:
        return {"state": UNTESTED, "reason": "not run on this case"}
    if obs.status != OK:
        return {"state": UNTESTED, "action_id": obs.action_id, "reason": f"{obs.status}: {obs.error or ''}".strip()}
    values = _numeric(obs.per_case().get(cid) or {})
    if evidence.step.analyzer == "prompt_contrast":
        values.update(_contrast_values(obs, case, answer_fn or default_answer))
    if not values:
        return {"state": UNTESTED, "action_id": obs.action_id, "reason": obs.missing.get(cid, "no per_case row")}
    return {"state": "measured", "action_id": obs.action_id, "values": values}


def follow_up_reading(evidence: StepEvidence, cases: Any, route_ids: "list[str]", control_ids: "list[str]",
                      answer_fn: "AnswerFn | None" = None) -> dict[str, Any]:
    """Mean of every numeric per-case reading over the route's FAIL cases and their PASS controls.

    Descriptive, like the S readings: no test, and a field is listed even when
    one cohort has no value for it. ``n_by_field`` counts the cases behind each
    mean (an answer-change reading needs both answers readable).
    """
    by_id = {str(c.id): c for c in cases}
    rows = {cid: follow_up_values(evidence, by_id[cid], answer_fn) for cid in [*route_ids, *control_ids]}
    out: dict[str, Any] = {}
    for name, ids in (("route_fail", route_ids), ("controls", control_ids)):
        measured = [rows[cid]["values"] for cid in ids if rows[cid]["state"] == "measured"]
        fields = sorted({k for v in measured for k in v})
        out[name] = {"n": len(ids), "n_measured": len(measured),
                     "means": {f: _mean([v[f] for v in measured if f in v]) for f in fields},
                     "n_by_field": {f: sum(1 for v in measured if f in v) for f in fields},
                     "untested": dict(Counter(rows[cid]["reason"] for cid in ids
                                              if rows[cid]["state"] != "measured"))}
    return out
