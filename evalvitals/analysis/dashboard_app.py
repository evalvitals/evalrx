"""Streamlit app for EvalVitals exploratory-analysis and diagnosis-loop runs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evalvitals.analysis import case_studio as cs
from evalvitals.analysis.dashboard import load_run
from evalvitals.reporting.stages import stage_specs_as_dicts
from evalvitals.viz.labels import display_name

# Eval-chart-style house theme (FAIL-red / PASS-slate palette, distribution-first
# chart builders, short names + number/bin formatting). See eval_viz_theme.py and
# the eval-chart-style SKILL for the chart-type policy this enforces.
try:
    from evalvitals.analysis import eval_viz_theme as viz
except Exception:  # plotly missing or import error — degrade to legacy rendering
    viz = None

try:
    from evalvitals.analysis.eval_case_matrix import continuous_signals, load_case_matrix
except Exception:
    load_case_matrix = None
    continuous_signals = None


def _viz_ready() -> bool:
    """Register the plotly template once per session; report whether viz is usable."""
    if viz is None:
        return False
    if not st.session_state.get("_viz_applied"):
        viz.apply()
        st.session_state["_viz_applied"] = True
    return True


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    session = load_run(run_arg)
    root = Path(session["root"])
    runs = session["runs"]

    st.set_page_config(
        page_title="EvalVitals",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None, "About": "EvalVitals diagnosis dashboard."},
    )
    _inject_css()
    _viz_ready()

    selected = _render_sidebar(root, session)

    if session.get("kind") == "loop" and session.get("story"):
        _render_loop_story(root, session["story"], runs)
        return

    if session.get("kind") == "casebench":
        render_case_study_run(root, session, selected)
        return

    if not runs:
        _render_empty(root)
        return

    render_explore_report(root, runs[selected])


def render_explore_report(root: Path, turn: dict[str, Any]) -> None:
    """Render one standalone explore report with the FIXED five-tab layout.

    Every explore-shaped result — a plain M2/M3 run, a held-out pipeline run,
    an uploaded-zip run — shows the same five tabs, so the reader never has to
    re-learn the page. Stages the run did not reach (held-out verdicts, fix)
    render as greyed "not available" placeholders instead of disappearing;
    they fill in the moment a downstream phase drops ``confirm_report.json`` /
    ``fix_report.json`` next to the exploratory report.

    Shared entry: dashboard_app's explore view and the upload workbench
    (upload_app.py) both render finished runs through this."""
    turn_dir = Path(turn["dir"])
    report = turn["report"]

    confirm = _load_sibling_json(turn_dir, "confirm_report.json")
    fix_report = _load_sibling_json(turn_dir, "fix_report.json")
    _render_report_overview(root, turn, report, turn_dir, confirm=confirm, fix_report=fix_report)

    tabs = st.tabs(EXPLORE_TAB_LABELS)
    with tabs[0]:
        _render_problem_setting(root, report, story=None, artifact_dir=turn_dir)
    with tabs[1]:
        _render_standalone_analysis(report, turn_dir, root)
    with tabs[2]:
        # Tab 3 stays the PURE proposal view (same as a plain explore run);
        # verdicts live in their own tab so proposal and validation never blur.
        _render_standalone_hypotheses(report, turn_dir)
    with tabs[3]:
        if confirm:
            _render_holdout_panel(confirm)
        else:
            _render_unavailable_panel(
                "Held-out verdicts",
                "This run stopped at M3: hypotheses were proposed but not "
                "re-tested on held-out data.",
                "A held-out confirm phase (pipeline phase 2 — frozen-recipe "
                "re-evaluation + e-BH + LLM judge, e.g. `test_hypotheses.py`) "
                "writes `confirm_report.json` next to the exploratory report; "
                "this panel then fills in.",
            )
    with tabs[4]:
        if fix_report:
            _render_fix_panel(fix_report)
        else:
            _render_unavailable_panel(
                "Fix",
                "No repair phase was run for this analysis.",
                "The surgery/fix phase (pipeline phase 3 — M5 confirm → M4 "
                "surgery → tiered fix L1–L3b, e.g. `run_surgery.py`) writes "
                "`fix_report.json` next to the exploratory report; this panel "
                "then fills in.",
            )


# One layout for every explore-shaped result; unavailable stages grey out
# rather than vanish (see render_explore_report).
EXPLORE_TAB_LABELS = [
    "1 Problem Setting", "2 Exploratory Analysis", "3 Hypotheses",
    "4 Validation results", "5 Fix",
]

# A paper-method bench run (examples/**/run.py) — see case_studio.py. Same
# numbered-tab grammar as the explore layout so the reader doesn't re-learn
# the page, but the middle tab is a per-case stimulus browser rather than a
# chart gallery: the benchmark is audio/image, so a human has to be able to
# play the clip and answer the question before judging a repair.
CASEBENCH_TAB_LABELS = ["1 Run Overview", "2 Case Study", "3 Repair Methods"]

#: Container MIME types Streamlit's <audio> element needs spelled out.
_AUDIO_MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac",
}

_FLIP_PILL = {
    cs.FLIP_REPAIRED: ("ev-pill-ok", "repaired this case"),
    cs.FLIP_BROKE: ("ev-pill-fail", "broke this case"),
    cs.FLIP_UNCHANGED: ("ev-pill", "no change"),
    cs.FLIP_UNTESTED: ("ev-pill", "not in the sweep"),
}


def render_case_study_run(root: Path, session: dict[str, Any], selected: int = 0) -> None:
    """Render one paper-method bench report as a playable case book.

    The report records outcomes per case id; :mod:`case_studio` joins those to
    the benchmark manifest so this view can show the actual stimulus. Three
    tabs: what the run did, the cases themselves (play / answer / compare), and
    what each repair candidate actually was."""
    studies = session.get("case_studies") or []
    if not studies:
        _render_empty(root)
        return
    study = studies[min(max(selected, 0), len(studies) - 1)]

    _render_casebench_header(root, study)
    for note in study.notes:
        st.caption(f"⚠︎ {note}")

    tabs = st.tabs(CASEBENCH_TAB_LABELS)
    with tabs[0]:
        _render_casebench_overview(study)
    with tabs[1]:
        _render_case_study_tab(study)
    with tabs[2]:
        _render_repair_methods_tab(study)


def _render_casebench_header(root: Path, study: Any) -> None:
    report = study.report
    paper = str(report.get("paper") or study.name)
    st.markdown(
        f"""
        <div class="ev-header">
          <div class="ev-kicker">Paper-method bench run</div>
          <h1>{_html_escape(paper.upper())} · {_html_escape(str(report.get('model', '?')))}</h1>
          <div class="ev-path">{_html_escape(str(study.report_path))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fidelity = str(report.get("paper_method_fidelity") or "—")
    splits = report.get("splits") or {}
    baseline = report.get("baseline") or {}
    tiles = [
        ("Cases with a stimulus", _format_int(sum(1 for c in study.cases if not c.unresolved)),
         f"{study.media_kind} benchmark"),
        ("Baseline accuracy · selection", _pct(_split_accuracy(baseline, "selection")),
         f"n={_format_int((splits or {}).get('selection'))}"),
        ("Baseline accuracy · confirmation", _pct(_split_accuracy(baseline, "confirmation")),
         f"n={_format_int((splits or {}).get('confirmation'))}"),
        ("Repair candidates tried", _format_int(len(study.candidates)),
         f"max tier {report.get('max_tier', '—')}"),
        ("Method fidelity", fidelity.replace("_", " "), str(report.get("backend", ""))),
    ]
    cols = st.columns(len(tiles))
    for col, (label, value, caption) in zip(cols, tiles):
        with col:
            st.markdown(
                f"""
                <div class="ev-metric-card">
                  <div class="ev-metric-label">{_html_escape(label)}</div>
                  <div class="ev-metric-value">{_html_escape(str(value))}</div>
                  <div class="ev-metric-caption">{_html_escape(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _split_accuracy(baseline: dict[str, Any], split: str) -> float | None:
    block = baseline.get(split)
    if not isinstance(block, dict):
        return None
    value = block.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _render_casebench_overview(study: Any) -> None:
    report = study.report
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">What this run tested</div>'
        '<div class="ev-section-sub">A frozen baseline probe, a diagnosis/selection split, a '
        "repair sweep validated pairwise on the selection split, and (when a candidate wins) a "
        "held-out confirmation pass.</div></div>",
        unsafe_allow_html=True,
    )

    hypothesis = report.get("hypothesis") or {}
    statement = str(hypothesis.get("statement") or "")
    if statement:
        st.markdown(
            f"""
            <div class="ev-report-answer">
              <div class="ev-brief-label">Hypothesis under test
                 · {_html_escape(str(hypothesis.get('predicted_failure_mode', '')))}</div>
              <div class="ev-report-answer-text">{_html_escape(statement)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    splits = report.get("splits") or {}
    baseline = report.get("baseline") or {}
    st.dataframe(pd.DataFrame([
        {
            "split": name,
            "n": (baseline.get(name) or {}).get("n", splits.get(name)),
            "baseline correct": (baseline.get(name) or {}).get("correct"),
            "baseline accuracy": _split_accuracy(baseline, name),
            "used for": {
                "diagnosis": "forming the hypothesis",
                "selection": "choosing a repair (the sweep runs here)",
                "confirmation": "held-out re-test of the winner",
            }[name],
        }
        for name in cs.SPLITS if baseline.get(name) or splits.get(name)
    ]), width="stretch", hide_index=True)

    confirmation = (report.get("auto_fix") or {}).get("confirmation") or {}
    if confirmation.get("skipped"):
        st.info(
            f"Confirmation was skipped — {confirmation['skipped']}. No candidate cleared "
            "selection, so nothing was re-tested on held-out data and this run has no "
            "confirmed repair."
        )
    elif confirmation:
        st.markdown("#### Held-out confirmation")
        st.dataframe(pd.DataFrame([confirmation]), width="stretch", hide_index=True)

    scored = [c for c in study.cases if c.baseline_correct is not None and c.metadata]
    keys = cs.facet_keys(scored)
    if keys:
        st.markdown("#### Where the baseline fails")
        key = st.selectbox("Break accuracy down by", keys, key=f"ev_cs_facet_break::{study.name}")
        rows = cs.accuracy_by_facet(scored, key)
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Baseline arm only, pooled over every split that recorded per-case rows — a "
            "descriptive breakdown of this run's own probe, not a benchmark result."
        )

    recommendation = ((report.get("auto_fix") or {}).get("selection") or {}).get("recommendation")
    if isinstance(recommendation, dict):
        st.markdown(
            f"""
            <div class="ev-report-answer">
              <div class="ev-brief-label">Recommendation</div>
              <div class="ev-report-answer-text">Escalate to
                {_html_escape(str(recommendation.get('recommend_tier', '')))} —
                {_html_escape(str(recommendation.get('reason', '')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Raw report JSON", expanded=False):
        st.json(report)


# ---------------------------------------------------------------------------
# Case study tab
# ---------------------------------------------------------------------------

_OUTCOME_FILTERS = {
    "all": "Every case",
    "baseline_wrong": "Baseline answered wrong",
    "baseline_right": "Baseline answered right",
    "repaired": "Repaired by some candidate",
    "broke": "Broken by some candidate",
    "flipped": "Flipped either way",
}


def _case_filter_keys(study: Any) -> dict[str, str]:
    """Session-state keys for this study's filter widgets (kept per report so
    two reports open in one session don't fight over one filter)."""
    return {
        "case": f"ev_cs_case::{study.name}",
        "split": f"ev_cs_split::{study.name}",
        "outcome": f"ev_cs_outcome::{study.name}",
        "blind": f"ev_cs_blind::{study.name}",
    }


def _jump_to_case(study: Any, case_id: str) -> None:
    """Select *case_id* in the Case Study tab and clear the filters that could
    hide it — used by the Repair Methods tab's per-case buttons. Runs as a
    widget callback, i.e. before the next script run builds the widgets."""
    keys = _case_filter_keys(study)
    st.session_state[keys["case"]] = case_id
    st.session_state[keys["split"]] = "all"
    st.session_state[keys["outcome"]] = "all"


def _filter_cases(cases: list[Any], *, split: str, outcome: str) -> list[Any]:
    out = [c for c in cases if split in ("all", c.split)]
    if outcome == "baseline_wrong":
        out = [c for c in out if c.baseline_correct is False]
    elif outcome == "baseline_right":
        out = [c for c in out if c.baseline_correct is True]
    elif outcome == "repaired":
        out = [c for c in out if c.flipped_by()]
    elif outcome == "broke":
        out = [c for c in out if c.broken_by()]
    elif outcome == "flipped":
        out = [c for c in out if c.flipped_by() or c.broken_by()]
    return out


def _render_case_study_tab(study: Any) -> None:
    keys = _case_filter_keys(study)
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Case study — play it, answer it, then see what happened'
        "</div>"
        '<div class="ev-section-sub">Every case the run scored, with its actual stimulus. '
        "Answer first with blind mode on if you want your own read of the item before the "
        "model's answer and the repair verdicts are revealed.</div></div>",
        unsafe_allow_html=True,
    )

    controls = st.columns([1.1, 1.2, 1.6])
    blind = controls[0].toggle(
        "Blind mode", value=True, key=keys["blind"],
        help="Hide the correct answer, the model's answer and the repair verdicts "
             "until you lock in your own answer.",
    )
    splits_present = [s for s in cs.SPLITS if any(c.split == s for c in study.cases)]
    split = controls[1].selectbox(
        "Split", ["all", *splits_present], key=keys["split"],
        format_func=lambda s: "All splits" if s == "all" else s,
    )
    # Filtering by outcome would leak the answer, so blind mode drops every
    # option but "all" — and clears a stale selection first, since a session
    # value outside the option list is undefined behaviour for the widget.
    outcome_options = list(_OUTCOME_FILTERS) if not blind else ["all"]
    if st.session_state.get(keys["outcome"]) not in outcome_options:
        st.session_state[keys["outcome"]] = "all"
    outcome = controls[2].selectbox(
        "Outcome filter", outcome_options, key=keys["outcome"],
        format_func=lambda k: _OUTCOME_FILTERS[k],
        help="Disabled in blind mode — filtering by outcome would leak the answer.",
    )

    cases = _filter_cases(study.cases, split=split, outcome=outcome)
    if not cases:
        st.info("No case matches these filters.")
        return

    ids = [c.id for c in cases]
    if st.session_state.get(keys["case"]) not in ids:
        st.session_state[keys["case"]] = ids[0]
    by_id = {c.id: c for c in cases}
    st.selectbox(
        f"Case ({len(ids)} shown)", ids, key=keys["case"],
        format_func=lambda cid: cs.case_label(by_id[cid], blind=blind),
    )
    case = by_id[st.session_state[keys["case"]]]

    nav = st.columns([1, 1, 6])
    index = ids.index(case.id)
    if nav[0].button("← Prev", disabled=index == 0, key=f"ev_cs_prev::{study.name}"):
        st.session_state[keys["case"]] = ids[index - 1]
        st.rerun()
    if nav[1].button("Next →", disabled=index == len(ids) - 1, key=f"ev_cs_next::{study.name}"):
        st.session_state[keys["case"]] = ids[index + 1]
        st.rerun()

    _render_case_panel(study, case, blind=blind)
    _render_human_scoreboard(study)


def _render_case_panel(study: Any, case: Any, *, blind: bool) -> None:
    answers = st.session_state.setdefault(f"ev_cs_answers::{study.name}", {})
    answered = case.id in answers
    revealed = answered or not blind

    media_col, qa_col = st.columns([1, 1.35])
    with media_col:
        _render_case_media(case)
    with qa_col:
        _render_case_question(study, case, answers=answers, revealed=revealed)

    if revealed:
        _render_case_outcomes(study, case, human=answers.get(case.id))
    else:
        st.caption("Lock in an answer (or switch blind mode off) to reveal the model's answer "
                   "and what each repair did to this case.")


def _render_case_media(case: Any) -> None:
    media = case.media
    if media.kind == "none":
        st.info("This case has no media file in the manifest — it is text-only, or the "
                "manifest row is missing.")
    elif not media.exists:
        st.warning(f"Media file not found: `{media.declared or '?'}`")
    elif media.kind == "audio":
        path = Path(media.path)
        try:
            st.audio(path.read_bytes(),
                     format=_AUDIO_MIME.get(path.suffix.lower(), "audio/wav"))
        except OSError as exc:
            st.warning(f"Could not read the audio file: {exc}")
    elif media.kind == "image":
        st.image(media.path, width="stretch")

    facts = []
    if case.duration_sec is not None:
        facts.append(f"{case.duration_sec:.1f}s")
    facts.extend(f"{k}: {v}" for k, v in (case.metadata or {}).items()
                 if not isinstance(v, (dict, list)))
    if facts:
        st.caption(" · ".join(str(f) for f in facts))
    st.caption(f"`{case.id}` · {case.split} split")


def _render_case_question(study: Any, case: Any, *, answers: dict[str, str], revealed: bool) -> None:
    if case.unresolved:
        st.warning("No manifest row for this case id — outcomes only, no question text.")
        return
    if not case.question:
        st.caption("The manifest row carries no question field.")
        return

    st.markdown(f"**{_html_escape(case.question)}**", unsafe_allow_html=True)
    if not case.choices:
        st.caption("Open-ended case (no options in the manifest).")
        return

    letters = [letter for letter, _ in case.choices]
    labels = {letter: f"{letter}. {text}" for letter, text in case.choices}
    picked_key = f"ev_cs_pick::{study.name}::{case.id}"
    current = answers.get(case.id)
    st.radio(
        "Your answer", letters, key=picked_key,
        index=letters.index(current) if current in letters else None,
        format_func=lambda letter: labels[letter],
    )

    buttons = st.columns([1, 1, 3])
    if buttons[0].button("Lock in", key=f"ev_cs_lock::{study.name}::{case.id}"):
        chosen = st.session_state.get(picked_key)
        if chosen:
            answers[case.id] = chosen
            st.rerun()
        else:
            st.warning("Pick an option first.")
    if case.id in answers and buttons[1].button(
        "Clear", key=f"ev_cs_clear::{study.name}::{case.id}"
    ):
        answers.pop(case.id, None)
        st.rerun()

    if revealed and case.expected:
        human = answers.get(case.id)
        expected_line = f"Correct answer: **{case.expected}**"
        if case.expected_text:
            expected_line += f" — {case.expected_text}"
        st.markdown(expected_line)
        if human:
            mark = "✅ you got it" if human == case.expected else f"❌ you answered {human}"
            st.markdown(mark)


def _render_case_outcomes(study: Any, case: Any, *, human: str | None) -> None:
    st.markdown("#### What the model answered, and what each repair did")
    correctness = (
        "—" if case.baseline_correct is None
        else ("correct" if case.baseline_correct else "wrong")
    )
    pill = ("ev-pill-ok" if case.baseline_correct
            else "ev-pill-fail" if case.baseline_correct is False else "ev-pill")
    st.markdown(
        f"""
        <div class="ev-evidence-card">
          <div class="ev-evidence-card-top">
            <span class="ev-evidence-name">Baseline (unmodified decode)</span>
            <span class="ev-pill {pill}">{correctness}</span>
          </div>
          <div class="ev-evidence-summary">answered
            <code>{_html_escape(case.baseline_output or '(empty)')}</code>
            · expected <code>{_html_escape(case.expected or '?')}</code>
            {f"· you answered <code>{_html_escape(human)}</code>" if human else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not case.flips:
        st.caption(
            "No repair candidate was validated on this case — the sweep runs on the "
            "selection split only."
        )
        return

    notes = {c["name"]: c for c in study.candidates}
    rows = []
    for name, flip in case.flips.items():
        candidate = notes.get(name, {})
        rows.append({
            "candidate": name,
            "tier": candidate.get("tier", ""),
            "effect on this case": _FLIP_PILL.get(flip, ("", flip))[1],
            "verdict on the split": candidate.get("verdict", ""),
            "method": candidate.get("note", ""),
        })
    rows.sort(key=lambda r: (r["effect on this case"] == "no change", r["candidate"]))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "“no change” means the candidate scored this case the same way the baseline did; "
        "the report does not distinguish that from a case the candidate never applied to. "
        "Candidate answer text is not recorded — see the Repair Methods tab for what each "
        "candidate actually changes."
    )


def _render_human_scoreboard(study: Any) -> None:
    answers = st.session_state.get(f"ev_cs_answers::{study.name}") or {}
    if not answers:
        return
    graded = [(cid, pick) for cid, pick in answers.items() if study.case_by_id(cid)]
    if not graded:
        return
    human_correct = sum(
        1 for cid, pick in graded if pick == (study.case_by_id(cid).expected or "").upper()
    )
    model_correct = sum(
        1 for cid, _ in graded if study.case_by_id(cid).baseline_correct
    )
    cols = st.columns(3)
    cols[0].metric("You", f"{human_correct}/{len(graded)}")
    cols[1].metric("Model, same cases", f"{model_correct}/{len(graded)}")
    cols[2].metric("Gap", f"{human_correct - model_correct:+d}")
    st.caption(
        "Your answers live in this browser session only — they are never written back "
        "into the run's report."
    )


# ---------------------------------------------------------------------------
# Repair methods tab
# ---------------------------------------------------------------------------


def _render_repair_methods_tab(study: Any) -> None:
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Repair methods — what was actually changed</div>'
        '<div class="ev-section-sub">Every candidate the sweep tried, the mechanism behind '
        "it, its exact configuration, and the cases it flipped. Each candidate is validated "
        "pairwise against the unmodified baseline (McNemar + e-value, e-BH across the "
        "family).</div></div>",
        unsafe_allow_html=True,
    )
    selection = (study.report.get("auto_fix") or {}).get("selection") or {}
    if not study.candidates:
        _render_unavailable_panel(
            "Repair methods",
            "This run recorded no repair candidates.",
            "A bench run writes them under `auto_fix.selection.attempted`.",
        )
        return

    for line in _fix_narrative(selection):
        st.markdown(f"- {line}")

    best = str(selection.get("best") or "")
    st.dataframe(pd.DataFrame([{
        "": "🏆" if c["name"] == best else "",
        "tier": c["tier"],
        "candidate": c["name"],
        "kind": c["kind"],
        "source": c["source"],
        "repaired": c["n_fixed"],
        "broke": c["n_broken"],
        "pairs": c["n_pairs"],
        "effect": c["effect"],
        "e-value": c["e_value"],
        "verdict": c["verdict"],
    } for c in study.candidates]), width="stretch", hide_index=True)
    if not best:
        st.caption(
            "No `best`: no candidate was both individually significant and an e-BH "
            "survivor across the family, so nothing went on to confirmation."
        )

    for candidate in study.candidates:
        _render_candidate_card(study, candidate)


def _render_candidate_card(study: Any, candidate: dict[str, Any]) -> None:
    title = f"{candidate['tier']} · {candidate['name']} — {candidate['verdict'] or 'no verdict'}"
    with st.expander(title, expanded=False):
        if candidate["note"]:
            st.markdown(candidate["note"])
        st.caption(
            f"kind `{candidate['kind']}` · source `{candidate['source']}` · "
            f"{candidate['summary'] or 'no paired summary recorded'}"
        )
        if candidate["prompt_template"]:
            st.markdown("**Prompt template applied to every case**")
            st.code(candidate["prompt_template"], language="text")
        if candidate["knobs"]:
            st.markdown("**Configuration**")
            st.dataframe(
                pd.DataFrame([{"setting": k, "value": str(v)}
                              for k, v in candidate["knobs"].items()]),
                width="stretch", hide_index=True,
            )
        elif candidate["defaults_only"]:
            st.caption("Empty payload — the method runs at its published defaults.")

        for label, case_ids in (("Repaired", candidate["fixed_cases"]),
                                ("Broke", candidate["broken_cases"])):
            if not case_ids:
                continue
            st.markdown(f"**{label} {len(case_ids)} case(s)** — open one in the Case Study tab:")
            for chunk_start in range(0, len(case_ids), 4):
                chunk = case_ids[chunk_start:chunk_start + 4]
                for col, case_id in zip(st.columns(4), chunk):
                    case = study.case_by_id(case_id)
                    col.button(
                        _truncate(case.question or case_id, 42) if case else case_id,
                        key=f"ev_cs_jump::{study.name}::{candidate['name']}::{case_id}",
                        on_click=_jump_to_case, args=(study, case_id),
                        width="stretch",
                    )


def _render_unavailable_panel(title: str, what_happened: str, how_to_get_it: str) -> None:
    """Greyed placeholder for a pipeline stage this run never reached."""
    st.markdown(
        '<div class="ev-unavailable">'
        f"<div class='ev-unavailable-title'>⚪ {title} — not available for this run</div>"
        f"<p>{what_happened}</p>"
        f"<p class='ev-unavailable-hint'>{how_to_get_it}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar(root: Path, session: dict[str, Any]) -> int:
    runs = session["runs"]
    kind = session.get("kind", "explore")

    st.sidebar.markdown('<div class="ev-sidebar-title">EvalVitals</div>', unsafe_allow_html=True)
    st.sidebar.caption(root.name)
    st.sidebar.markdown(f"**Mode:** {kind}")

    if kind == "casebench":
        studies = session.get("case_studies") or []
        st.sidebar.markdown("---")
        selected = st.sidebar.radio(
            "Bench reports", range(len(studies)),
            format_func=lambda i: studies[i].name,
        ) if len(studies) > 1 else 0
        study = studies[int(selected)]
        st.sidebar.metric("Cases", len(study.cases))
        st.sidebar.metric("Repair candidates", len(study.candidates))
        st.sidebar.caption(
            f"Manifest: {study.manifest_path.name}" if study.manifest_path
            else "No benchmark manifest resolved"
        )
        return int(selected)

    if not runs:
        return 0

    if kind == "loop":
        st.sidebar.markdown("---")
        st.sidebar.caption("Diagnostic loop run — see the story view.")
        return 0

    labels = [_turn_label(t) for t in runs]
    selected = st.sidebar.radio("Reports", range(len(runs)), format_func=lambda i: labels[i])

    st.sidebar.markdown("---")
    st.sidebar.metric("Reports", len(runs))
    return int(selected)


def _render_empty(root: Path) -> None:
    st.markdown('<div class="ev-hero"><h1>EvalVitals Dashboard</h1></div>', unsafe_allow_html=True)
    st.warning(
        "No exploratory_report.json / fused_report.json / run_log.jsonl found, and no "
        "paper-method bench report (a JSON with `baseline` + `auto_fix`, e.g. an "
        "examples/**/outputs/*.json) either."
    )
    st.caption(str(root))


def _run_lifecycle(story: dict[str, Any]) -> dict[str, Any]:
    """Run-level provenance pulled from the additive run_start/loop_end story
    keys (see load_loop_story) — every field is optional, so callers must
    tolerate None throughout."""
    run_start = story.get("run_start") or {}
    loop_end = story.get("loop_end") or {}
    protocol = run_start.get("protocol") or {}
    label_dist = run_start.get("label_distribution") or {}
    return {
        "n_cases": run_start.get("n_cases"),
        "model": run_start.get("model"),
        "decision_judge": run_start.get("decision_judge"),
        "max_actions": run_start.get("max_actions"),
        "duration_sec": loop_end.get("total_duration_sec"),
        "resolved": loop_end.get("resolved"),
        "stopped_by": loop_end.get("stopped_by"),
        "n_verified": loop_end.get("n_verified"),
        "cycles": loop_end.get("cycles"),
        # What this run is actually investigating — logged at run_start but,
        # before this, never surfaced anywhere in the dashboard (a real gap:
        # a first-time viewer had no way to see what was being tested).
        "protocol_description": protocol.get("description") or "",
        "task_domain": protocol.get("task_domain") or "",
        "n_fail": label_dist.get("FAIL"),
        "n_pass": label_dist.get("PASS"),
    }


def _clean_repr(value: Any) -> str:
    """Strip a bare object repr's memory address (`" at 0x7f..."`) — a
    fallback repr() is much more readable without it."""
    import re

    return re.sub(r" at 0x[0-9a-fA-F]+", "", str(value))


def _hero_verdict(hyps: list[dict[str, Any]], lifecycle: dict[str, Any], *, descriptive: bool) -> tuple[str, str, str]:
    """Return (badge_text, pill_class, detail_text) for the hero band.

    Descriptive runs never claim a verdict (see _story_is_descriptive) — this
    states that honestly instead of guessing one from partial data."""
    if descriptive:
        return ("Analysis phase", "ev-pill", "No verdict yet — this run hasn't reached a test phase.")
    tested = [h for h in hyps if h.get("m5")]
    supported = [h for h in tested if h["m5"].get("status") == "supported"]
    if supported:
        return (
            "Resolved · Supported", "ev-pill-ok",
            f"Testing supported this explanation for the failures: “{supported[0]['statement']}”",
        )
    if tested and all(h["m5"].get("status") == "refuted" for h in tested):
        return ("Resolved · Refuted", "ev-pill-fail", "No hypothesis survived testing.")
    if tested:
        return ("Inconclusive", "ev-pill-warn", "Testing ran but did not clearly resolve a hypothesis.")
    stopped_by = lifecycle.get("stopped_by")
    if stopped_by:
        return (
            f"Stopped · {str(stopped_by).replace('_', ' ')}", "ev-pill",
            "The run stopped before reaching a tested hypothesis.",
        )
    return ("In progress", "ev-pill", "No hypothesis has been tested yet.")


def _render_hero_band(root: Path, story: dict[str, Any], *, descriptive: bool, hyps: list[dict[str, Any]], agentic: bool) -> None:
    """Answer-first hero band: mode badge, verdict pill + one-line conclusion,
    then a KPI stat-tile row — replaces the old raw "M1 -> M2 -> M3" header so
    the first screen states what happened instead of making a reader dig."""
    lifecycle = _run_lifecycle(story)
    if agentic:
        kicker = "Agentic Diagnosis Run"
    elif descriptive:
        kicker = "Analysis Phase"
    else:
        kicker = "Diagnostic Loop Run"
    badge_text, pill_class, detail = _hero_verdict(hyps, lifecycle, descriptive=descriptive)

    task_line = ""
    if lifecycle.get("protocol_description"):
        task_line = (
            f'<div class="ev-hero-task">Investigating: {_html_escape(_truncate(lifecycle["protocol_description"], 200))}</div>'
        )

    st.markdown(
        f"""
        <div class="ev-header">
          <div>
            <div class="ev-kicker">{_html_escape(kicker)}</div>
            {task_line}
            <div class="ev-hero-verdict">
              <span class="ev-pill {pill_class}">{_html_escape(badge_text)}</span>
            </div>
            <div class="ev-hero-verdict-text">{_html_escape(_truncate(detail, 220))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(str(root))

    tiles: list[tuple[str, str, str]] = []
    stages_run = ", ".join(sorted(_active_stages(story))) or "none yet"
    tiles.append(("Stages run", stages_run, ""))
    if hyps:
        tiles.append(("Hypotheses", _format_int(len(hyps)),
                       f"{lifecycle['n_verified']} verified" if lifecycle.get("n_verified") else "proposed"))
    if lifecycle.get("n_cases") is not None:
        case_caption = ""
        if lifecycle.get("n_fail") is not None and lifecycle.get("n_pass") is not None:
            case_caption = f"{lifecycle['n_fail']} FAIL / {lifecycle['n_pass']} PASS"
        tiles.append(("Cases", _format_int(lifecycle["n_cases"]), case_caption))
    if lifecycle.get("duration_sec") is not None:
        tiles.append(("Duration", f"{lifecycle['duration_sec']:.0f}s", ""))
    if agentic and lifecycle.get("max_actions") is not None:
        n_steps = len(story.get("agent_steps") or [])
        tiles.append(("Actions", f"{n_steps} / {lifecycle['max_actions']}", "steps taken / budget"))

    if tiles:
        # Single-line per tile: a multi-line/indented chunk here would trip
        # Markdown's "4-space indent = code block" rule once concatenated,
        # rendering everything past the first tile as literal escaped text.
        cells = "".join(
            f'<div class="ev-metric-card">'
            f'<div class="ev-metric-label">{_html_escape(label)}</div>'
            f'<div class="ev-metric-value">{_html_escape(value)}</div>'
            f'<div class="ev-metric-caption">{_html_escape(caption)}</div>'
            f"</div>"
            for label, value, caption in tiles
        )
        st.markdown(f'<div class="ev-kpi-row">{cells}</div>', unsafe_allow_html=True)

    secondary = []
    if lifecycle.get("model"):
        secondary.append(f"model: {_clean_repr(lifecycle['model'])}")
    if agentic and lifecycle.get("decision_judge"):
        secondary.append(f"judge: {_clean_repr(lifecycle['decision_judge'])}")
    if secondary:
        st.caption(" · ".join(secondary))


def _render_agent_trajectory_tab(story: dict[str, Any]) -> None:
    """The agentic decision timeline: what the judge chose, why, and whether
    the host accepted it. A rejected dispatch (stop-gate, precondition,
    max_actions) is rendered as a visible discipline feature, not an error —
    it's proof the loop can't just declare victory early."""
    steps = story.get("agent_steps") or []
    lifecycle = _run_lifecycle(story)

    budget_bits = []
    if lifecycle.get("max_actions") is not None:
        budget_bits.append(f"{len(steps)} / {lifecycle['max_actions']} steps")
    if lifecycle.get("stopped_by"):
        budget_bits.append(f"stopped by: {lifecycle['stopped_by']}")
    if lifecycle.get("duration_sec") is not None:
        budget_bits.append(f"{lifecycle['duration_sec']:.0f}s")
    if budget_bits:
        st.caption(" · ".join(budget_bits))

    if not steps:
        st.info("This run has no agentic decision steps recorded.", icon="ℹ️")
        return

    for step in steps:
        outcome = step.get("outcome")
        rejected = bool(outcome) and not outcome.get("ok", True)
        cls = "ev-step ev-step-rejected" if rejected else "ev-step"
        action = str(step.get("action") or "")
        rationale = str(step.get("rationale") or "")
        top_pill = ""
        if rejected:
            reason = str(outcome.get("error") or "rejected")
            top_pill = f'<span class="ev-pill ev-pill-warn">Rejected · {_html_escape(reason)}</span>'
        elif outcome is not None:
            top_pill = '<span class="ev-pill ev-pill-ok">Accepted</span>'
        flags = []
        if step.get("repair_attempts"):
            flags.append(f'<span class="ev-pill ev-pill-warn">{step["repair_attempts"]} repair attempt(s)</span>')
        if step.get("fallback_used"):
            flags.append('<span class="ev-pill ev-pill-warn">fallback used</span>')
        st.markdown(
            f"""
            <div class="{cls}">
              <div class="ev-step-top">
                <span class="ev-step-index">Step {step.get('step')}</span>
                {top_pill} {''.join(flags)}
              </div>
              <div class="ev-step-tool">{_html_escape(action)}</div>
              <div class="ev-step-rationale">{_html_escape(rationale)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if step.get("params"):
            st.code(json.dumps(step["params"], indent=2, default=str), language="json")
        if outcome:
            summary = str(outcome.get("summary") or outcome.get("error") or "")
            if summary:
                st.markdown(f'<div class="ev-step-outcome">{_html_escape(summary)}</div>', unsafe_allow_html=True)
        if step.get("judge_io"):
            with st.expander(f"Judge I/O — step {step.get('step')}", expanded=False):
                st.json(step["judge_io"])


def _render_loop_story(root: Path, story: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    """Render a diagnostic loop run: an answer-first hero band, then Problem
    Setting -> (Agent Trajectory, for agentic runs) -> Analysis -> Hypotheses."""
    # The Step-1 explore report (signals/charts/tables M2 confirmed & M3 consulted)
    # usually lives in a sibling dir; load_loop_story resolves it for us.
    explore_report = story.get("explore_report")
    if not explore_report:
        explore_report = next((r["report"] for r in runs if r["name"] == "fused_report"), None)
    explore_dir = Path(story.get("explore_dir") or (runs[0]["dir"] if runs else root))

    descriptive = _story_is_descriptive(story)
    hyps = _hypotheses_with_outcomes(story)
    agent_steps = story.get("agent_steps") or []

    _render_hero_band(root, story, descriptive=descriptive, hyps=hyps, agentic=bool(agent_steps))

    if not explore_report:
        st.info(
            "No explore report was found alongside this loop log, so measured "
            "signals/charts are unavailable. Re-run with `--explore-report`, or "
            "point the dashboard at a directory containing `fused_report.json` / "
            "`exploratory_report.json`.",
            icon="ℹ️",
        )

    has_downstream = bool(story.get("surgeries") or story.get("fixes"))
    hypothesis_label = "Hypotheses & Artifacts" if has_downstream else "Proposed Hypotheses"
    labels = ["Problem Setting"]
    if agent_steps:
        labels.append("Agent Trajectory")
    labels += ["Analysis", hypothesis_label]
    tabs = st.tabs([f"{i + 1} {label}" for i, label in enumerate(labels)])

    idx = 0
    with tabs[idx]:
        _render_problem_setting(root, explore_report or {}, story=story, artifact_dir=explore_dir)
    idx += 1
    if agent_steps:
        with tabs[idx]:
            _render_agent_trajectory_tab(story)
        idx += 1
    with tabs[idx]:
        _render_loop_analysis_panel(story, explore_report, explore_dir, root)
    idx += 1
    with tabs[idx]:
        _render_hypothesis_decision_panel(story, explore_report, explore_dir)


def _candidate_signals(explore_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [s for s in ((explore_report or {}).get("candidate_signals") or []) if isinstance(s, dict)]


def _story_is_descriptive(story: dict[str, Any] | None) -> bool:
    """True when the loaded run is analysis-phase only: every M2 deferred its
    validity verdict (``descriptive_only``) and no confirm/surgery has run yet.

    Mirrors ``reporting.compiler._is_descriptive_only`` so the analysis dashboard
    presents candidate signals + hypotheses WITHOUT a supported/not-supported
    verdict. The explorer's per-recipe ``reject`` flags are NOT confirmation; they
    are shown descriptively until the confirm phase adjudicates them."""
    if not story:
        return False
    analyses = story.get("analyses") or []
    if not analyses or not all(a.get("descriptive_only") for a in analyses):
        return False
    return not (story.get("surgeries") or [])


def _hypotheses_with_outcomes(story: dict[str, Any]) -> list[dict[str, Any]]:
    """Join each M3 hypothesis with the explore artifacts it cited and the
    M5/M4 tests that later evaluated it (matched by statement)."""
    tests_by_stmt: dict[str, list[dict[str, Any]]] = {}
    for s in story.get("surgeries") or []:
        tests_by_stmt.setdefault(str(s.get("hypothesis", "")).strip(), []).append(s)

    out: list[dict[str, Any]] = []
    for d in story.get("diagnoses") or []:
        for h in d.get("hypotheses") or []:
            stmt = _hypothesis_statement(h)
            if not stmt:
                continue
            tests = tests_by_stmt.get(stmt, [])
            m5_test = next((t for t in tests if t.get("module") == "m5"), None)
            m5 = None
            if m5_test is not None:
                evidence = m5_test.get("evidence") or {}
                m5 = {
                    "status": m5_test.get("status"),
                    "effect_size": evidence.get("m5_effect_size"),
                    "confidence": evidence.get("m5_confidence"),
                    "protocol_consistent": evidence.get("m5_protocol_consistent"),
                }
            out.append({
                "statement": stmt,
                "failure_mode": _hypothesis_failure_mode(h),
                "test_design": _hypothesis_test_design(h),
                "cycle": d.get("cycle"),
                "referenced_charts": d.get("referenced_charts") or [],
                "tests": tests,
                "m5": m5,
            })
    return out


def _hypothesis_statement(hypothesis: dict[str, Any]) -> str:
    return str(
        hypothesis.get("statement")
        or hypothesis.get("hypothesis")
        or hypothesis.get("predicate")
        or ""
    ).strip()


def _hypothesis_failure_mode(hypothesis: dict[str, Any]) -> str:
    return str(
        hypothesis.get("failure_mode")
        or hypothesis.get("predicted_failure_mode")
        or hypothesis.get("mode")
        or ""
    ).strip()


def _hypothesis_test_design(hypothesis: dict[str, Any]) -> str:
    """How M3 says this claim could be checked (the LLM's own TEST: line) —
    not a verdict, just the evidence/analyzer it named as relevant."""
    return str(hypothesis.get("test_design") or hypothesis.get("test") or "").strip()


def _render_problem_setting(
    root: Path,
    report: dict[str, Any] | None,
    *,
    story: dict[str, Any] | None,
    artifact_dir: Path | None = None,
) -> None:
    """Panel 1: orient the user before any statistical claims."""
    report = report or {}
    storyboard = _storyboard_panels(report, story=story)
    # The explore report's own "question" wins; a loop/agentic run without one
    # (no explore_report artifact) falls back to the protocol description
    # logged at run_start — otherwise a first-time viewer sees no explanation
    # of what this run is even investigating.
    fallback_question = "What distinguishes failures from passes?"
    if story:
        fallback_question = _run_lifecycle(story).get("protocol_description") or fallback_question
    question = str(report.get("question") or fallback_question)
    plain_question = str(report.get("plain_question") or "").strip()
    signals = _candidate_signals(report)
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]
    chart_count = (
        len(charts) + len(report.get("plots") or [])
        if "charts" in report or "plots" in report else None
    )

    matrix = None
    if load_case_matrix is not None:
        try:
            matrix = load_case_matrix(root)
        except Exception:
            matrix = None

    n_cases = None
    n_fail = None
    n_pass = None
    n_features = None
    if matrix is not None and not matrix.empty:
        n_cases = len(matrix)
        if "is_fail" in matrix.columns:
            n_fail = int(pd.to_numeric(matrix["is_fail"], errors="coerce").fillna(0).sum())
            n_pass = n_cases - n_fail
        try:
            n_features = len(continuous_signals(matrix)) if continuous_signals else None
        except Exception:
            n_features = None

    profile = report.get("data_profile") or {}
    if n_cases is None:
        n_cases = profile.get("loaded_rows", profile.get("n_rows"))
    columns = profile.get("columns") or {}
    if n_features is None and isinstance(columns, dict):
        n_features = len(columns)
    split = report.get("split") or {}
    if n_cases is None:
        n_cases = split.get("n_total")
    n_explore = split.get("n_explore")
    n_confirm = split.get("n_confirm")
    if n_fail is None or n_pass is None:
        cb = _class_balance_from_report(report, artifact_dir)
        if cb:
            n_fail = cb.get("FAIL", n_fail)
            n_pass = cb.get("PASS", n_pass)

    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Problem Setting</div>'
        '<div class="ev-section-sub">What this run studied, what one row means, '
        "and how to interpret the analysis.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if story:
        _render_stage_map(active={"M1"})
    _render_storyboard_panel(storyboard, "problem_setting")
    if not _has_storyboard_panel(storyboard, "problem_setting"):
        question_technical_line = (
            f'<div class="ev-path">Original question: {_html_escape(question)}</div>'
            if plain_question and question.strip() and question.strip() != plain_question
            else ""
        )
        st.markdown(
            f"""
            <div class="ev-report-answer">
              <div class="ev-brief-label">User question</div>
              <div class="ev-report-answer-text">{_html_escape(plain_question or question)}</div>
              {question_technical_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

    metrics = [
        ("Cases", n_cases, "rows/cases loaded"),
        ("Explore", n_explore, "agent discovery split"),
        ("Confirm", n_confirm, "held-out confirmation split"),
        ("Candidate patterns", len(signals) or n_features, "candidate/features"),
        ("Charts", chart_count, "analysis visuals"),
    ]
    metrics = [(label, value, help_text) for label, value, help_text in metrics if value is not None]
    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, value, help_text) in zip(cols, metrics, strict=False):
            col.metric(label, _format_int(value), help=help_text)

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.markdown("#### What data was provided")
        if n_fail is not None or n_pass is not None:
            st.markdown(
                f"""
                <div class="ev-brief-grid ev-brief-grid-two">
                  <div class="ev-brief-card">
                    <div class="ev-brief-label">FAIL cases</div>
                    <div class="ev-metric-value">{_html_escape(_format_int(n_fail))}</div>
                  </div>
                  <div class="ev-brief-card">
                    <div class="ev-brief-label">PASS cases</div>
                    <div class="ev-metric-value">{_html_escape(_format_int(n_pass))}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if matrix is not None and not matrix.empty:
            fields = [
                c for c in matrix.columns
                if c not in {"case_id", "label", "is_fail", "model_yes", "truth_yes"}
            ]
            st.caption(
                "Dashboard reconstructed the per-case feature matrix. "
                "The analysis below compares these per-case signals against FAIL/PASS labels."
                if story else
                "Reconstructed per-case feature matrix from the sampled input records."
            )
            preview = pd.DataFrame({
                "field": fields[:12],
                "display": [display_name(f) for f in fields[:12]],
                "coverage": [int(matrix[f].notna().sum()) for f in fields[:12]],
            })
            if not preview.empty:
                st.dataframe(preview, width="stretch", hide_index=True, height=260)
        elif isinstance(columns, dict) and columns:
            st.caption("Explorer data profile from the sampled input records.")
            schema = _column_schema_dataframe(columns)
            if not schema.empty:
                st.dataframe(schema, width="stretch", hide_index=True, height=260)
        else:
            st.info("No structured data profile was saved with this report.")

    with c2:
        st.markdown("#### How to interpret this analysis")
        # Standalone explore reports have no loop/story at all, so they're
        # always the descriptive framing; a loop run's own story says whether
        # a test phase actually ran.
        descriptive = _story_is_descriptive(story) if story else True
        stages = []
        if story:
            stages = [
                ("M2 analyses", len(story.get("analyses") or [])),
                ("M3 diagnoses", len(story.get("diagnoses") or [])),
            ]
            if not descriptive:
                surgeries = story.get("surgeries") or []
                stages.append(("M4/M5 tests", len(surgeries)))
                stages.append(("Fixes", len(story.get("fixes") or [])))
        method_text = (
            "Exploratory (descriptive) — no confirm/test phase has run yet for this hypothesis"
            if descriptive else
            "Confirmatory — a test phase ran and hypotheses below carry validity verdicts"
        )
        st.markdown(
            f"""
            <div class="ev-brief-card">
              <div class="ev-brief-label">Analysis method</div>
              <div class="ev-brief-value">{_html_escape(method_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, count in stages:
            st.markdown(f"- **{label}:** {count}")
        # The run directory is already one click away under the header details;
        # repeating it here put an absolute home path in the middle of the
        # page, where any screenshot or screen-share carries it along.


def _render_stage_map(*, active: set[str]) -> None:
    """Small stage strip: the full M1-M5 pipeline map, with the stages this
    run actually reached highlighted (see :func:`_active_stages`)."""
    chips = []
    for spec in stage_specs_as_dicts():
        cls = "ev-stage-chip ev-stage-active" if spec["id"] in active else "ev-stage-chip"
        chips.append(
            f'<span class="{cls}"><b>{_html_escape(spec["id"])}</b> '
            f'{_html_escape(spec["name"])}</span>'
        )
    st.markdown('<div class="ev-stage-strip">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def _active_stages(story: dict[str, Any] | None) -> set[str]:
    """Which M1-M5 stages this run's own log actually reached, for the stage
    strip's highlighting — a run that stopped at M3 shows M4/M5 as pipeline
    context, not as if they'd run."""
    story = story or {}
    active: set[str] = set()
    if story.get("probes"):
        active.add("M1")
    if story.get("analyses"):
        active.add("M2")
    if story.get("diagnoses"):
        active.add("M3")
    surgeries = story.get("surgeries") or []
    if any(s.get("module") == "m4" for s in surgeries):
        active.add("M4")
    if any(s.get("module") == "m5" for s in surgeries):
        active.add("M5")
    return active


def _class_balance_from_report(
    report: dict[str, Any] | None,
    artifact_dir: Path | None,
) -> dict[str, int]:
    if not report or artifact_dir is None:
        return {}
    for chart in report.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        name = str(chart.get("name") or "").lower()
        if "class_balance" not in name:
            continue
        df = _table_to_dataframe(chart.get("data"), artifact_dir)
        if df is None or df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        outcome_col = cols.get("outcome") or df.columns[0]
        count_col = cols.get("count")
        if count_col is None:
            continue
        out: dict[str, int] = {}
        for _, row in df.iterrows():
            key = str(row[outcome_col]).upper()
            try:
                out[key] = int(row[count_col])
            except Exception:
                pass
        return out
    return {}


def _storyboard_panels(
    report: dict[str, Any] | None,
    *,
    story: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Agent-authored dashboard storyboard, with compiled diagnostic fallback."""
    report = report or {}
    raw = report.get("dashboard_storyboard") or report.get("ui_panels")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]
    diag = (story or {}).get("diagnostic_report") or {}
    raw = diag.get("dashboard_storyboard")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]
    return []


def _has_storyboard_panel(panels: list[dict[str, Any]], panel_id: str) -> bool:
    return any(str(p.get("id")) == panel_id for p in panels)


def _render_storyboard_panel(panels: list[dict[str, Any]], panel_id: str) -> None:
    panel = next((p for p in panels if str(p.get("id")) == panel_id), None)
    if not panel:
        return
    stages = ", ".join(str(s) for s in (panel.get("stages") or []))
    items = [_humanize_storyboard_text(str(x)) for x in (panel.get("items") or []) if str(x).strip()]
    refs = [str(x) for x in (panel.get("artifact_refs") or []) if str(x).strip()]
    st.markdown(
        f"""
        <div class="ev-storyboard-card">
          <div class="ev-brief-label">Run-specific narrative {("· " + _html_escape(stages)) if stages else ""}</div>
          <div class="ev-storyboard-title">{_html_escape(str(panel.get('title') or panel_id))}</div>
          <div class="ev-storyboard-summary">{_html_escape(_humanize_storyboard_text(str(panel.get('summary') or '')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if items:
        st.markdown("**Storyboard takeaways**")
        for item in items[:8]:
            st.markdown(f"- {item}")
    if refs:
        st.caption("Storyboard artifact refs: " + ", ".join(refs[:8]))


def _humanize_storyboard_text(text: str) -> str:
    replacements = {
        "generated_probe1_false_detection": "sanity-check probe false-detection flag",
        "probe1_false_detection": "sanity-check probe false-detection flag",
        "relative_attention_max_relative_weight": "maximum relative attention",
        "relative_attention_mean_relative_weight": "mean relative attention",
        "relative_attention_focus_share": "attention focus share",
        "low_focus_share": "high attention focus",
        "probe1_positive": "probe positive flag",
    }
    out = str(text)
    for raw, label in replacements.items():
        out = out.replace(raw, label)
    return out


def _analysis_takeaway(
    report: dict[str, Any] | None,
    story: dict[str, Any] | None = None,
    *,
    descriptive: bool = True,
) -> str:
    candidates = [s for s in _candidate_signals(report) if not _is_leaky_signal(s)]
    if not descriptive:
        # compile_diagnostic_report's `answer` is itself a confirmatory verdict
        # (a "supported" claim's text, or an explicit "no supported claim"
        # fallback — see reporting/compiler.py::_answer) — only trust it in
        # non-descriptive mode; the descriptive branch below never asks it.
        diag = (story or {}).get("diagnostic_report") or {}
        answer = str(diag.get("answer") or "").strip()
        if answer:
            return answer
    if descriptive:
        if candidates:
            def _abs_eff(s: dict[str, Any]) -> float:
                try:
                    return abs(float(s.get("effect")))
                except (TypeError, ValueError):
                    return 0.0
            lead = max(candidates, key=_abs_eff)
            name = display_name(lead.get("display_name") or lead.get("name"))
            return f"{name} shows the largest FAIL-vs-PASS separation in this run (descriptive; not yet confirmed)."
        conclusion = str((report or {}).get("conclusion") or "").strip()
        return conclusion or "No candidate signals were identified in the loaded report."
    supported = [
        display_name(s.get("display_name") or s.get("name"))
        for s in candidates
        if s.get("reject") is True
    ]
    if supported:
        return f"{supported[0]} is the leading held-out supported signal."
    conclusion = str((report or {}).get("conclusion") or "").strip()
    return conclusion or "No supported diagnostic claim is available in the loaded report."


def _render_method_card(title: str, method: str, evidence: str, takeaway: str) -> None:
    st.markdown(
        f"""
        <div class="ev-analysis-card">
          <div class="ev-brief-label">{_html_escape(title)}</div>
          <div class="ev-analysis-method">Method: {_html_escape(method)}</div>
          <div class="ev-analysis-evidence">Evidence: {_html_escape(evidence)}</div>
          <div class="ev-analysis-takeaway">Takeaway: {_html_escape(takeaway)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_loop_analysis_panel(story, explore_report, explore_dir, root=None) -> None:
    """Panel 2: analysis methods, evidence, charts, and takeaways."""
    explore_report = explore_report or {}
    if not explore_report:
        st.info(
            "No explore report was found alongside this loop log, so the Analysis "
            "panel cannot show measured signals or charts. Re-run with "
            "`--explore-report`, or point the dashboard at a directory containing "
            "`fused_report.json` / `exploratory_report.json`.",
            icon="ℹ️",
        )
    signals = _candidate_signals(explore_report)
    adj = explore_report.get("adjudication") or {}
    readings = [r for r in explore_report.get("chart_readings") or [] if isinstance(r, dict)]
    storyboard = _storyboard_panels(explore_report, story=story)
    # Descriptive (no verdict yet) vs confirmatory (surgery/fix data exists) —
    # see _story_is_descriptive; every branch below already keys off this.
    descriptive = _story_is_descriptive(story)

    st.markdown("### Analysis")
    _render_stage_map(active=_active_stages(story))
    if descriptive:
        st.info(
            "Analysis phase — candidate signals and proposed hypotheses are shown "
            "**descriptively, without a validity verdict**. A confirm/test phase "
            "hasn't run yet for this hypothesis.",
            icon="🔍",
        )
    else:
        st.info(
            "Confirmatory — signals and hypotheses below carry validity verdicts "
            "from the test phase that ran for this hypothesis.",
            icon="✅",
        )
    st.markdown(
        f"""
        <div class="ev-report-answer">
          <div class="ev-brief-label">Bottom line</div>
          <div class="ev-report-answer-text">{_html_escape(_analysis_takeaway(explore_report, story, descriptive=descriptive))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if signals:
        st.markdown("#### Candidate signals" if descriptive else "#### Evidence you can use")
        _render_confirmatory_evidence(signals, explore_report, explore_dir, descriptive=descriptive)

    with st.expander("Method details and generated storyboard", expanded=False):
        _render_storyboard_panel(storyboard, "analysis")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if descriptive:
                _render_method_card(
                    "Descriptive signal screening",
                    "host-computed effect sizes + CIs (no validity verdict)",
                    "Effect sizes and confidence intervals only — e-BH/FDR adjudication is deferred.",
                    f"{len(signals)} candidate signal(s) described; none adjudicated yet "
                    "(run the confirm phase to test them).",
                )
            else:
                _render_method_card(
                    "Confirmatory signal testing",
                    str(adj.get("method") or "held-out host adjudication"),
                    "Effect sizes, confidence intervals, and e-BH/FDR verdicts.",
                    f"{adj.get('n_signals_rejected', adj.get('n_rejected', 0))} of "
                    f"{adj.get('n_signals_tested', adj.get('n_in_family', len(signals)))} tested signals survived.",
                )
        with c2:
            _render_method_card(
                "Exploratory visualization",
                "Agent-proposed chart plan; host-rendered deterministic specs.",
                f"{len(readings)} chart reading(s), {len(explore_report.get('charts') or [])} chart spec(s).",
                readings[0].get("reading", "Charts are leads for hypotheses, not causal proof.")
                if readings else "No chart readings were recorded.",
            )
        # e-BH adjudication numbers are a confirmatory verdict — only shown once
        # M2 is no longer forced into descriptive-only mode (see `descriptive` above).
        if adj and not descriptive:
            st.dataframe(pd.DataFrame([{
                "method": adj.get("method", "-"),
                "alpha": adj.get("alpha", "-"),
                "signals tested": adj.get("n_signals_tested", adj.get("n_in_family", "-")),
                "rejected": adj.get("n_signals_rejected", adj.get("n_rejected", "-")),
                "split": adj.get("split", "-"),
            }]), width="stretch", hide_index=True)

    with st.expander("Extra diagnostics and raw tables", expanded=False):
        _render_visual_plan(explore_report)
        if readings:
            rows = [{
                "chart": display_name(r.get("chart")),
                "takeaway": r.get("reading", ""),
                "do not infer": r.get("do_not_infer", ""),
            } for r in readings]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if _viz_ready():
            _render_explore_tables(explore_report, explore_dir, full=False, root=root)
        else:
            _render_charts_and_plots(explore_report, explore_dir)
        _render_stat_panel(root)


def _render_hypothesis_decision_panel(story, explore_report, explore_dir) -> None:
    """Panel 3: M3 hypotheses, with M4/M5 verdicts joined in when this run's
    own data has them (see :func:`_hypotheses_with_outcomes`'s ``m5`` key)."""
    descriptive = _story_is_descriptive(story)
    hyps = _hypotheses_with_outcomes(story)
    storyboard = _storyboard_panels(explore_report or {}, story=story)
    st.markdown("### Proposed Hypotheses")
    _render_stage_map(active=_active_stages(story))
    _render_storyboard_panel(storyboard, "hypotheses_artifacts")
    if hyps:
        if descriptive:
            st.caption("These are M3 hypotheses proposed from the M2 analysis. A "
                       "confirm/test phase hasn't run yet for this hypothesis.")
        else:
            st.caption("M3 hypotheses joined with their M4/M5 test verdicts, where tested.")
        for h in hyps:
            _render_hypothesis_card(h, descriptive=descriptive)
    else:
        st.warning("No M3 hypotheses were recorded for this run.")

    with st.expander("Inspect M1-M5 artifacts and flow", expanded=True):
        _render_loop_flow(story, explore_report, explore_dir)
    with st.expander("Inspect raw tables", expanded=False):
        _render_explore_tables(explore_report, explore_dir, full=True)
    if explore_report:
        with st.expander("Explorer original charts/figures", expanded=False):
            _render_charts_and_plots(explore_report, explore_dir)


def _column_schema_dataframe(columns: dict[str, Any], *, limit: int | None = None) -> pd.DataFrame:
    """Turn a ``DatasetProfile.columns`` dict into a readable schema table.

    Shared by the Problem Setting and Exploratory Analysis tabs so a reader
    sees the same column/role/type breakdown regardless of which tab they
    land on first — this is the one place that has to adapt per folder, since
    every run's schema is different."""
    rows = []
    items = list(columns.items())
    if limit is not None:
        items = items[:limit]
    for name, col in items:
        if not isinstance(col, dict):
            continue
        lo, hi = col.get("numeric_min"), col.get("numeric_max")
        value_range = f"{lo:g} – {hi:g}" if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else "—"
        rows.append({
            "Role": str(col.get("role", "predictor")),
            "Field": display_name(str(name)),
            "Type": str(col.get("dtype", "")),
            "Unique": col.get("unique", 0),
            "Missing": col.get("missing", 0),
            "Range": value_range,
        })
    if not rows:
        return pd.DataFrame()
    role_order = {"outcome": 0, "id": 1, "group": 2, "time": 3, "predictor": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda d: d["Role"].map(role_order).fillna(9))
        .sort_values(["_order", "Field"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _render_folder_scan(scan: dict[str, Any]) -> None:
    """What the agent literally found on disk before any row-level parsing:
    file/dir counts, extension mix, how many JSON files were sampled, and the
    raw relative-path listing. This differs folder to folder and isn't
    captured by the parsed row/column schema below it."""
    root = str(scan.get("root") or "")
    n_files = scan.get("n_files_total", 0)
    n_dirs = scan.get("n_dirs", 0)
    json_found = scan.get("json_files_found", 0)
    json_used = scan.get("json_files_used", 0)
    extensions = scan.get("extensions") or {}
    entries = [str(e) for e in scan.get("entries") or []]
    truncated = bool(scan.get("truncated"))

    ext_pills = "".join(
        f'<span class="ev-pill">{_html_escape(str(ext))} · {_html_escape(_format_int(count))}</span>'
        for ext, count in list(extensions.items())[:10]
    )
    sampling_note = (
        f"{json_used} of {json_found} JSON file(s) sampled"
        if json_found != json_used
        else f"{json_used} JSON file(s)"
    )
    summary_line = (
        f"{_format_int(n_files)} files across {_format_int(n_dirs)} "
        f"subdirector{'y' if n_dirs == 1 else 'ies'} · {sampling_note}"
    )
    st.markdown(
        f"""
        <div class="ev-structure-card">
          <div class="ev-brief-label">Folder contents · {_html_escape(root)}</div>
          <div class="ev-structure-outcome">{_html_escape(summary_line)}</div>
          <div style="margin-top:0.55rem; display:flex; flex-wrap:wrap; gap:0.35rem;">{ext_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if scan.get("scan_capped"):
        st.caption(
            "This folder is large enough that the scan stopped early — counts above "
            "are a lower bound, not the full total."
        )
    if entries:
        label = "Folder listing" + (f" (first {len(entries)} entries)" if truncated else "")
        with st.expander(label, expanded=True):
            st.code("\n".join(entries), language=None)


def _render_data_structure_panel(report: dict[str, Any]) -> None:
    """Orient on schema before findings: every folder's data looks different,
    so the analysis tab states what was found on disk, then row count, outcome
    kind, and the full column/role/type breakdown — up front, instead of
    assuming a fixed shape."""
    profile = report.get("data_profile") or {}
    columns = profile.get("columns") or {}
    scan = profile.get("folder_scan") or {}
    if scan:
        _render_folder_scan(scan)

    if not isinstance(columns, dict) or not columns:
        return

    outcome = profile.get("outcome") or {}
    grain = str(profile.get("grain") or "unknown")
    n_rows = profile.get("loaded_rows", profile.get("n_rows"))
    kind = str(outcome.get("kind", "none"))
    if kind == "none" or not outcome.get("present"):
        outcome_text = "No outcome column detected — this is unsupervised exploration."
    else:
        kind_label = {"binary": "Binary", "categorical": "Categorical", "continuous": "Continuous"}.get(
            kind, kind.title()
        )
        unique = outcome.get("unique", 0)
        outcome_text = (
            f"{kind_label} outcome: {display_name(str(outcome.get('column')))} "
            f"({unique} distinct value{'s' if unique != 1 else ''})"
        )

    st.markdown(
        f"""
        <div class="ev-structure-card">
          <div class="ev-brief-label">
            Data structure · {_html_escape(grain)}-level · {_html_escape(_format_int(n_rows))} rows profiled
            · {_html_escape(_format_int(len(columns)))} columns
          </div>
          <div class="ev-structure-outcome">{_html_escape(outcome_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    schema = _column_schema_dataframe(columns)
    if not schema.empty:
        st.dataframe(
            schema, width="stretch", hide_index=True, height=min(360, 46 + 35 * len(schema))
        )

    warnings = [str(w) for w in profile.get("warnings") or []]
    if warnings:
        st.caption("Profiling notes: " + "; ".join(warnings[:5]))


def _render_raw_data_browser(report: dict[str, Any], turn_dir: Path) -> None:
    """Let a reader browse the actual rows the agent analyzed, not just their
    schema — when a takeaway looks surprising, the next thing people want is
    to see the underlying records themselves, searchable, right here."""
    df: pd.DataFrame | None = None
    note = ""
    records_path = _resolve_artifact_path("records.json", turn_dir)
    if records_path.exists():
        try:
            data = json.loads(records_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
        except (OSError, json.JSONDecodeError, ValueError):
            df = None
    if df is None:
        sample_rows = (report.get("data_profile") or {}).get("sample_rows") or []
        if sample_rows:
            df = pd.DataFrame(sample_rows)
            note = (
                f"Only a small sample was saved with this report ({len(sample_rows)} row(s)); "
                "the full loaded records weren't persisted alongside it."
            )
    if df is None or df.empty:
        return

    with st.expander(f"Browse raw data ({len(df)} row(s) loaded)", expanded=False):
        if note:
            st.caption(note)
        search = st.text_input(
            "Search rows", value="", placeholder="Filter by any value…", key="raw_data_search"
        )
        view = df
        if search.strip():
            mask = df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False, regex=False)
            ).any(axis=1)
            view = df[mask]
        shown = min(len(view), 500)
        st.caption(f"Showing {shown} of {len(view)} matching row(s) (of {len(df)} total).")
        st.dataframe(view.head(500), width="stretch", height=420)


def _takeaway_headline(takeaway: dict[str, Any]) -> str:
    plain_title = str(takeaway.get("plain_title") or "").strip()
    title = str(takeaway.get("title") or "").strip()
    return plain_title or title or "Finding"


def _render_findings_overview(takeaways: list[dict[str, Any]]) -> None:
    if not takeaways:
        return

    items = []
    for i, takeaway in enumerate(takeaways, start=1):
        chart_count = len(takeaway.get("chart_names") or [])
        table_count = len(takeaway.get("table_names") or [])
        meta = [
            f"{chart_count} visual{'s' if chart_count != 1 else ''}",
            f"{table_count} source table{'s' if table_count != 1 else ''}",
        ]
        if takeaway.get("caveat"):
            meta.append("caution noted")
        if takeaway.get("analysis"):
            meta.append("detail available")
        items.append(
            '<div class="ev-finding-index-row">'
            f'<div class="ev-finding-index-num">{i}</div>'
            '<div>'
            f'<div class="ev-finding-index-title">{_html_escape(_takeaway_headline(takeaway))}</div>'
            f'<div class="ev-finding-index-meta">{_html_escape(" · ".join(meta))}</div>'
            '</div>'
            '</div>'
        )
    st.markdown(
        '<div class="ev-findings-overview">'
        '<div class="ev-findings-overview-kicker">Findings overview</div>'
        '<div class="ev-findings-overview-sub">'
        'Skim the conclusions first; open each finding’s details for the full analysis and source data.'
        '</div>'
        f'{"".join(items)}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_takeaway_details(
    index: int,
    takeaway: dict[str, Any],
    *,
    title: str,
    headline: str,
    table_names: list[str],
    tables: dict[str, Any],
    turn_dir: Path,
    referenced_tables: set[str],
) -> None:
    has_technical_title = bool(title.strip() and title.strip() != headline.strip())
    has_detail = bool(
        has_technical_title
        or takeaway.get("analysis")
        or takeaway.get("caveat")
        or any(name in tables for name in table_names)
    )
    if not has_detail:
        return

    with st.expander(f"Details and source data for finding {index}", expanded=False):
        if has_technical_title:
            st.markdown("**Technical title**")
            st.markdown(_html_escape(title))
        if takeaway.get("analysis"):
            st.markdown("**Full analysis**")
            st.markdown(_html_escape(str(takeaway["analysis"])))
        if takeaway.get("caveat"):
            st.markdown("**Caveat**")
            st.markdown(_html_escape(str(takeaway["caveat"])))
        for name in table_names:
            referenced_tables.add(name)
            source = tables.get(name)
            if source is None:
                continue
            df = _table_to_dataframe(source, turn_dir)
            if df is not None:
                with st.expander(f"Source table: {display_name(name)}", expanded=False):
                    st.dataframe(df, width="stretch", height=220)


def _render_standalone_analysis(report: dict[str, Any], turn_dir: Path, root: Path) -> None:
    """The primary exploratory-analysis view: pure descriptive EDA, no hypotheses.

    Each takeaway is rendered as a finding -> supporting visual evidence ->
    compact interpretation, with full analysis and source data available on
    demand so the page stays scannable."""
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Exploratory Analysis</div>'
        '<div class="ev-section-sub">Descriptive findings only — skim the evidence cards first, '
        "then open details for full analysis, caveats, and source data.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_data_structure_panel(report)
    _render_raw_data_browser(report, turn_dir)
    storyboard = _storyboard_panels(report, story=None)
    _render_storyboard_panel(storyboard, "problem_setting")

    observations = [str(o) for o in (report.get("observations") or [])]
    if observations:
        with st.expander("Data overview", expanded=False):
            for obs in observations:
                st.markdown(f"- {obs}")

    caveats = [str(c) for c in (report.get("caveats") or [])]
    if caveats:
        with st.expander("Caveats", expanded=False):
            for caveat in caveats:
                st.markdown(f"- {caveat}")

    takeaways = [
        t for t in report.get("takeaways") or []
        if isinstance(t, dict) and (t.get("title") or t.get("plain_title"))
    ]
    if not takeaways:
        st.info(
            "No structured takeaways were recorded for this report — showing the "
            "raw charts and tables instead."
        )
        _render_charts_and_plots(report, turn_dir)
        _render_tables(report, turn_dir)
        return

    charts_by_name = _chart_lookup(report)
    plots_by_stem = _plot_lookup(report)
    tables = report.get("tables") or {}
    referenced_charts: set[str] = set()
    referenced_tables: set[str] = set()

    _render_findings_overview(takeaways)

    for i, takeaway in enumerate(takeaways, start=1):
        with st.container(border=True):
            title = str(takeaway.get("title", ""))
            headline = _takeaway_headline(takeaway)
            st.markdown(
                '<div class="ev-takeaway-head">'
                f'<div class="ev-takeaway-badge">{i}</div>'
                f'<div class="ev-takeaway-title">{_html_escape(headline)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            chart_names = [str(x) for x in takeaway.get("chart_names") or []]
            table_names = [str(x) for x in takeaway.get("table_names") or []]
            found_charts = []
            for name in chart_names:
                referenced_charts.add(name)
                if name in charts_by_name:
                    found_charts.append((name, "chart", charts_by_name[name]))
                elif name in plots_by_stem:
                    found_charts.append((name, "plot", plots_by_stem[name]))
            if found_charts:
                # Always a two-column grid: a takeaway with a single chart gets a
                # half-width card, not a full-page blow-up — keeps every
                # takeaway's evidence at the same visual scale.
                cols = st.columns(2)
                for idx, (artifact_name, kind, item) in enumerate(found_charts):
                    with cols[idx % 2]:
                        if kind == "chart":
                            _render_chart_card(item, turn_dir, heading_level="caption", key_prefix=f"takeaway{i}")
                            _render_compact_visual_explanation(report, name=artifact_name, chart=item)
                        else:
                            _render_plot_card(item, turn_dir)
                            _render_compact_visual_explanation(report, name=artifact_name)
            if chart_names or table_names:
                if not found_charts and not any(n in tables for n in table_names):
                    st.markdown(
                        '<div class="ev-takeaway-evidence-empty">'
                        "(referenced evidence not found among this report's artifacts)"
                        "</div>",
                        unsafe_allow_html=True,
                    )
            if takeaway.get("analysis") and not found_charts:
                st.markdown(
                    f'<div class="ev-takeaway-analysis">{_html_escape(str(takeaway["analysis"]))}</div>',
                    unsafe_allow_html=True,
                )
            _render_takeaway_details(
                i,
                takeaway,
                title=title,
                headline=headline,
                table_names=table_names,
                tables=tables,
                turn_dir=turn_dir,
                referenced_tables=referenced_tables,
            )

    orphan_charts = [c for name, c in charts_by_name.items() if name not in referenced_charts]
    orphan_plots = [p for stem, p in plots_by_stem.items() if stem not in referenced_charts]
    orphan_tables = {k: v for k, v in tables.items() if k not in referenced_tables}
    if orphan_charts or orphan_plots or orphan_tables:
        _render_unlinked_exploratory_material(
            report, turn_dir, orphan_charts, orphan_plots, orphan_tables
        )

    signals = _candidate_signals(report)
    with st.expander("Run details", expanded=False):
        st.caption(
            f"Agent-authored EDA plus host-rendered deterministic chart specs. "
            f"{len(signals)} candidate signal(s), {len(report.get('charts') or [])} chart spec(s), "
            f"{len(takeaways)} takeaway(s)."
        )
        _render_visual_plan(report)


def _render_unlinked_exploratory_material(
    report: dict[str, Any],
    turn_dir: Path,
    charts: list[dict[str, Any]],
    plots: list[Any],
    tables: dict[str, Any],
) -> None:
    """Show uncited artifacts as an explicit QA review, never as conclusions.

    The explorer generates a broad visual inventory before ranking takeaways.
    This keeps that audit trail visible, while making the missing link or the
    agent-recorded supporting purpose obvious to the reader.
    """
    plan = {
        str(item.get("name")): item
        for item in report.get("visual_plan") or []
        if isinstance(item, dict) and item.get("name")
    }
    n_artifacts = len(charts) + len(plots) + len(tables)
    with st.expander(
        f"Supporting exploratory material (not a conclusion) — {n_artifacts}",
        expanded=False,
    ):
        st.caption(
            "These artifacts were generated during broad exploration but were not cited as evidence for a ranked takeaway. "
            "Treat them as context or a QA item, not as an additional finding."
        )
        st.caption(
            "For new runs, the reason for retaining a supporting visual is recorded below. "
            "If an item says its linkage is missing, ask a follow-up to analyze or promote it explicitly."
        )

        def _context(name: str) -> None:
            item = plan.get(name)
            if item is None:
                st.info(
                    "Linkage missing: this artifact has no matching visualization-plan entry, "
                    "so the report does not explain why it was retained."
                )
                return
            question = str(item.get("question") or "an exploratory question")
            rationale = str(item.get("rationale") or "")
            disposition = str(item.get("disposition") or "").lower()
            reason = str(item.get("not_promoted_reason") or "")
            if disposition == "primary":
                st.warning(
                    "Linkage missing: the analysis plan marked this as a primary visual, "
                    "but no takeaway cites it. Review it before treating it as a conclusion."
                )
            elif disposition == "supporting":
                st.caption(
                    "Supporting purpose — " + (reason or "context/diagnostic evidence; not promoted to a ranked takeaway.")
                )
            else:
                st.caption(
                    "The report did not classify this visual as primary or supporting; it is shown for audit only."
                )
            details = f"It was designed to answer: {question}."
            if rationale:
                details += f" Why this form: {rationale}"
            st.caption(details)

        # Unreferenced visuals still belong on the same grid as everything
        # else — rendering them full-bleed made an audit-only chart the largest
        # thing on the page.
        for start in range(0, len(charts), 2):
            orphan_cols = st.columns(2)
            for offset, chart in enumerate(charts[start:start + 2]):
                with orphan_cols[offset]:
                    idx = start + offset
                    name = str(chart.get("name") or "")
                    st.markdown(f"**{display_name(name) if name else 'Unnamed chart'}**")
                    _context(name)
                    _render_chart_card(
                        chart, turn_dir, heading_level="caption", key_prefix=f"orphan_chart_{idx}"
                    )
                    _render_visual_explanation(report, name=name, chart=chart)
        for start in range(0, len(plots), 2):
            plot_cols = st.columns(2)
            for offset, path in enumerate(plots[start:start + 2]):
                with plot_cols[offset]:
                    # ``_plot_lookup`` deliberately retains the report's
                    # serialized string path. Do not assume a Path here:
                    # agent-authored reports may also contain a Path-like
                    # object or another JSON scalar.
                    name = Path(str(path)).stem
                    st.markdown(f"**{display_name(name)}**")
                    _context(name)
                    _render_plot_card(path, turn_dir)
                    _render_visual_explanation(report, name=name)
        for name, source in tables.items():
            st.markdown(f"**{display_name(name)}**")
            _context(name)
            df = _table_to_dataframe(source, turn_dir)
            if df is not None:
                st.dataframe(df, width="stretch", height=220)


def _visual_key(value: Any) -> str:
    """Loose but deterministic matching for chart names, titles, and PNG stems."""
    raw = Path(str(value or "")).stem.lower()
    return "".join(ch for ch in raw if ch.isalnum())


def _chart_reading_for(
    report: dict[str, Any], *, name: str, chart: dict[str, Any] | None = None
) -> tuple[str | None, str | None]:
    keys = {_visual_key(name)}
    if chart is not None:
        keys.update(
            _visual_key(chart.get(field))
            for field in ("name", "title", "display_name", "figure_path")
        )
    keys.discard("")
    for reading in report.get("chart_readings") or []:
        if not isinstance(reading, dict):
            continue
        if _visual_key(reading.get("chart")) in keys:
            text = str(reading.get("reading") or "").strip() or None
            boundary = str(reading.get("do_not_infer") or "").strip() or None
            return text, boundary
    return None, None


def _render_compact_visual_explanation(
    report: dict[str, Any], *, name: str, chart: dict[str, Any] | None = None
) -> None:
    reading, boundary = _chart_reading_for(report, name=name, chart=chart)
    if reading:
        st.markdown(
            f'<div class="ev-evidence-reading">'
            f'<div class="ev-evidence-reading-label">Evidence reading</div>'
            f'<div>{_html_escape(reading)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif chart is not None and chart.get("x") and chart.get("y"):
        kind = str(chart.get("kind") or "chart").lower()
        st.markdown(
            f'<div class="ev-evidence-reading">'
            f'<div class="ev-evidence-reading-label">Visual guide</div>'
            f'<div>This {kind} encodes <code>{_html_escape(str(chart["y"]))}</code> '
            f'against <code>{_html_escape(str(chart["x"]))}</code>. '
            'The dashboard is not assigning a specific finding to this visual.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Supplementary visual: read its axes, legend, and annotations before drawing a conclusion."
        )
    if boundary:
        st.markdown(
            f'<div class="ev-evidence-caution">'
            f'<span>Caution</span> {_html_escape(boundary)}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_visual_explanation(
    report: dict[str, Any], *, name: str, chart: dict[str, Any] | None = None
) -> None:
    """Give every rendered visual an interpretation and a boundary on inference.

    Agent-authored readings are the authoritative interpretation. Older reports
    sometimes lack them, so we still orient the reader to the chart's encoding
    without inventing a data finding.
    """
    st.markdown("**How to read this visual**")
    reading, boundary = _chart_reading_for(report, name=name, chart=chart)
    if reading:
        st.markdown(f"**What it shows:** {reading}")
        if boundary:
            st.caption(f"Do not infer: {boundary}")
        return

    # Structural fallback: useful for legacy artifacts, but deliberately does
    # not claim a pattern the agent did not record.
    if chart is not None and chart.get("x") and chart.get("y"):
        kind = str(chart.get("kind") or "chart").lower()
        st.markdown(
            f"**What it shows:** This {kind} encodes `{chart['y']}` against `{chart['x']}`. "
            "Compare levels, trends, or spread across the displayed values."
        )
    else:
        st.markdown(
            "**What it shows:** This is a supplementary visual from the exploratory pass; "
            "read its axes, legend, and annotations before drawing a conclusion."
        )
    st.caption(
        "No agent-authored reading was saved for this visual, so the dashboard is not assigning it a finding."
    )


def _load_sibling_json(turn_dir: Path, name: str) -> dict[str, Any] | None:
    """Read a pipeline artifact JSON living next to the exploratory report;
    None when absent or unreadable (the tab simply doesn't appear)."""
    import json

    path = Path(turn_dir) / name
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


_VERDICT_COLORS = {
    "supported": "var(--ev-ok)",
    "partial": "var(--ev-warn)",
    "refuted": "var(--ev-fail)",
    "not_testable": "var(--ev-muted)",
    "not_judged": "var(--ev-muted)",
}


def _render_standalone_hypotheses(report: dict[str, Any], turn_dir: Path) -> None:
    """Panel 3: M3 hypotheses proposed from M2's takeaways — the PURE proposal
    view (identical whether or not a downstream confirm phase ran; held-out
    verdicts get their own tab). Candidate signals / suggested next steps /
    raw artifacts are demoted into an expander below, since they're optional
    inputs for a separate downstream pipeline, not the primary content of
    this tab."""
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Hypotheses</div>'
        '<div class="ev-section-sub">M3 — falsifiable candidate explanations proposed from the '
        "M2 takeaways above. Proposed only, not validated here — held-out "
        "verdicts, if any, live in their own tab.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    hypotheses = [h for h in report.get("hypotheses") or [] if isinstance(h, dict) and h.get("statement")]
    if hypotheses:
        for h in hypotheses:
            _render_standalone_hypothesis_card(h)
    else:
        st.info("No formal hypotheses were recorded for this report.")
        st.markdown(
            "Review the candidate signals below as possible follow-ups, or rerun the "
            "analysis with hypothesis generation enabled."
        )

    signals = _candidate_signals(report)
    tests = report.get("recommended_confirmatory_tests") or []
    with st.expander("Candidate signals — possible follow-ups, not validated hypotheses", expanded=False):
        if signals:
            st.markdown("#### Candidate signals — possible follow-ups, not validated hypotheses")
            st.dataframe(_signals_dataframe(signals), width="stretch", hide_index=True)
        if tests:
            st.markdown("#### Suggested next steps")
            for item in tests:
                st.markdown(f"- {item}")
        if not signals and not tests:
            st.caption("No candidate signals or suggested next steps were recorded.")
    _render_artifacts(report, turn_dir)


def _render_standalone_hypothesis_card(
    h: dict[str, Any], verdict: dict[str, Any] | None = None
) -> None:
    statement = str(h.get("statement") or "").strip()
    plain_statement = str(h.get("plain_statement") or "").strip()
    headline = plain_statement or statement
    technical_line = (
        f'<div class="ev-signal-test">Technical detail: {_html_escape(statement)}</div>'
        if plain_statement and statement and statement != plain_statement else ""
    )
    basis = str(h.get("basis") or "").strip()
    test_design = str(h.get("test_design") or "").strip()
    basis_line = f'<div class="ev-signal-body">based on: {_html_escape(basis)}</div>' if basis else ""
    test_line = (
        f'<div class="ev-signal-test">How this could be checked: {_html_escape(test_design)}</div>'
        if test_design else ""
    )
    badge = reasoning_line = ""
    if verdict:
        v = str(verdict.get("verdict") or "not_judged")
        color = _VERDICT_COLORS.get(v, "var(--ev-muted)")
        surgery = " · routed to surgery" if verdict.get("needs_surgery") else ""
        badge = (
            f'<span class="ev-pill" style="border-color:{color};color:{color};">'
            f"held-out: {_html_escape(v)}</span>"
        )
        reasoning = str(verdict.get("reasoning") or "").strip()
        if reasoning:
            reasoning_line = (
                f'<div class="ev-signal-test">Judge: {_html_escape(reasoning)}'
                f"{_html_escape(surgery)}</div>"
            )
    st.markdown(
        f"""
        <div class="ev-signal">
          <div class="ev-signal-title">{_html_escape(headline)} {badge}</div>
          {technical_line}
          {basis_line}
          {test_line}
          {reasoning_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_holdout_panel(confirm: dict[str, Any]) -> None:
    """Validation results tab: phase 2's frozen-recipe re-adjudication + the
    per-hypothesis judge verdicts. Unlike the in-sample screen in tab 2,
    verdicts here are legitimate: thresholds were frozen before this data was
    touched."""
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Validation results</div>'
        '<div class="ev-section-sub">Recipes re-evaluated VERBATIM (thresholds frozen from the '
        "exploration half) on a validate split the explorer never saw; hypotheses graded "
        "against that evidence.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    adj = confirm.get("adjudication") or {}
    cols = st.columns(4)
    cols[0].metric("Validate cases", _format_int(confirm.get("n_validate_rows")))
    cols[1].metric("FAIL in validate", _format_int(confirm.get("n_validate_fail")))
    cols[2].metric("Signals adjudicated", _format_int(adj.get("n_host_adjudicated")))
    cols[3].metric("Held-out rejections", _format_int(adj.get("n_rejected")))
    st.caption(
        f"Adjudication: {adj.get('method', 'e-BH')} at alpha={adj.get('alpha', '?')}, "
        f"split={adj.get('split', 'held_out')} — a REJECT here is a real held-out "
        "verdict, not the exploration half's in-sample screen."
    )

    rows = []
    for s in confirm.get("signal_verdicts") or []:
        if not isinstance(s, dict):
            continue
        rows.append({
            "signal": display_name(s.get("display_name") or s.get("name")),
            "status": s.get("status"),
            "held-out verdict": ("REJECT H0" if s.get("reject") else "not rejected")
            if s.get("status") == "adjudicated" else (s.get("reason") or "—"),
            "fail rate (flagged)": s.get("fail_rate_flagged"),
            "fail rate (unflagged)": s.get("fail_rate_unflagged"),
            "effect": s.get("effect"),
            "CI": str(s.get("ci")) if s.get("ci") is not None else "—",
            "n": s.get("n_holdout"),
        })
    if rows:
        st.markdown("#### Signal recipes on the held-out split")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    hyps = [v for v in confirm.get("hypothesis_verdicts") or [] if isinstance(v, dict)]
    if hyps:
        st.markdown("#### Hypothesis verdicts")
        judge = confirm.get("judge") or {}
        if judge:
            st.caption(f"Graded by {judge.get('model', 'LLM judge')} against the held-out table.")
        for v in hyps:
            _render_standalone_hypothesis_card(
                {k: v.get(k) for k in ("statement", "basis", "test_design")}, verdict=v
            )


def _fmt_e(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _fix_narrative(fix: dict[str, Any]) -> list[str]:
    """Deterministic reader-facing digest of the repair sweep: the winner, the
    family-corrected survivors, a prompt-level vs internals-write contrast when
    the data shows one, and the refine signal. Built from the structured
    `attempted` list so it stays faithful to the paired-statistics record."""
    attempted = [a for a in fix.get("attempted") or [] if isinstance(a, dict)]
    if not attempted:
        return []
    lines: list[str] = []
    by_name = {str(a.get("name")): a for a in attempted}

    best = by_name.get(str(fix.get("best")))
    if best:
        outcome = "REJECT H0 [fixed]" if best.get("reject") else str(best.get("verdict"))
        lines.append(
            f"🏆 **Winner — {best.get('tier')} `{best.get('name')}`**: repaired "
            f"{best.get('n_fixed')} case(s), broke {best.get('n_broken')}, "
            f"e-value {_fmt_e(best.get('e_value'))} → {outcome}."
        )

    survivors = [str(n) for n in fix.get("ebh_survivors") or []]
    if survivors:
        parts = []
        for name in survivors:
            a = by_name.get(name)
            parts.append(
                f"`{name}` ({a.get('tier')}: {a.get('n_fixed')} fixed / "
                f"{a.get('n_broken')} broken, e={_fmt_e(a.get('e_value'))})"
                if a else f"`{name}`"
            )
        lines.append("Survived e-BH across the whole candidate family: " + "; ".join(parts) + ".")

    def _group_best(prefixes: tuple[str, ...]):
        group = [a for a in attempted if str(a.get("tier", "")).startswith(prefixes)]
        return max(group, key=lambda a: float(a.get("e_value") or 0.0), default=None)

    prompt_best = _group_best(("L1", "L2"))
    internals_best = _group_best(("L3",))
    if prompt_best and internals_best:
        if prompt_best.get("reject") and not internals_best.get("reject"):
            lines.append(
                f"Prompt/input-level repairs (L1/L2) beat internals-write interventions: the "
                f"best L3 candidate `{internals_best.get('name')}` reached only "
                f"e={_fmt_e(internals_best.get('e_value'))} ({internals_best.get('verdict')}) vs "
                f"e={_fmt_e(prompt_best.get('e_value'))} for `{prompt_best.get('name')}`."
            )
        elif internals_best.get("reject") and not prompt_best.get("reject"):
            lines.append(
                f"Internals-write interventions beat prompt-level repairs: "
                f"`{internals_best.get('name')}` e={_fmt_e(internals_best.get('e_value'))} vs best "
                f"prompt candidate e={_fmt_e(prompt_best.get('e_value'))}."
            )

    refine = fix.get("refine_signal")
    if isinstance(refine, dict) and refine.get("message"):
        lines.append(f"Refine signal: {refine['message']}")
    return lines


def _render_fix_panel(fix_report: dict[str, Any]) -> None:
    """Fix tab: phase 3's surgery context (M5/M4) and the tiered repair sweep —
    what was attempted at each tier, with the paired McNemar / e-value verdict
    per candidate, summarized for the reader before the full table."""
    st.markdown(
        '<div class="ev-section-head">'
        '<div class="ev-section-title">Fix — surgery &amp; tiered repair</div>'
        '<div class="ev-section-sub">Repair candidates from prompt templates (L1) through '
        "input specs (L2) to internals writes (L3a/L3b), each validated pairwise against "
        "the unmodified baseline (McNemar + e-value; e-BH across the family; no-free-lunch "
        "on present-object probes).</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Model {fix_report.get('model', '?')} · {_format_int(fix_report.get('n_cases'))} "
        f"loop cases · full logs: {fix_report.get('logs', '—')}"
    )

    m5 = [m for m in fix_report.get("m5_results") or [] if isinstance(m, dict)]
    if m5:
        st.markdown("#### Surgery context — M5 confirmation")
        st.dataframe(pd.DataFrame([{
            "hypothesis": _truncate(str(m.get("statement", "")), 110),
            "M5 status": m.get("status"),
            "confidence": m.get("confidence"),
            "grade": m.get("evidence_grade"),
            "held-out verdict": m.get("holdout_verdict"),
        } for m in m5]), width="stretch", hide_index=True)

    fix = fix_report.get("fix") or {}
    narrative = _fix_narrative(fix)
    if narrative:
        st.markdown("#### What was repaired, and how well")
        for line in narrative:
            st.markdown(f"- {line}")

    attempted = [a for a in fix.get("attempted") or [] if isinstance(a, dict)]
    if attempted:
        best_name = str(fix.get("best"))
        rows = [{
            "": "🏆" if str(a.get("name")) == best_name else "",
            "tier": a.get("tier"),
            "candidate": a.get("name"),
            "kind": a.get("kind"),
            "repaired": a.get("n_fixed"),
            "broke": a.get("n_broken"),
            "coverage": a.get("coverage"),
            "e-value": a.get("e_value"),
            "verdict": a.get("verdict"),
        } for a in attempted]
        rows.sort(key=lambda r: (str(r["tier"]), -float(r["e-value"] or 0.0)))
        st.markdown("#### All repair candidates")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            '"repaired/broke" are paired flips vs the unmodified baseline over the same '
            "cases; a candidate is `fixed` only when its e-value survives e-BH across "
            "every candidate tried (the correction for best-of-N selection)."
        )

    rec = fix.get("recommendation")
    if rec:
        rec_text = (
            f"escalate to {rec.get('recommend_tier')}: {rec.get('reason', '')}"
            if isinstance(rec, dict) else str(rec)
        )
        st.markdown(
            f"""
            <div class="ev-report-answer">
              <div class="ev-brief-label">Recommendation</div>
              <div class="ev-report-answer-text">{_html_escape(rec_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Raw M4 / fix records", expanded=False):
        if fix_report.get("m4"):
            st.markdown("**M4 surgery**")
            st.code(str(fix_report["m4"])[:2000])
        if fix:
            st.json(fix)


def _render_stat_panel(root) -> None:
    """Inferential / distributional panel built from the per-case feature matrix
    reconstructed from m1_state.pkl: model-evaluation, distribution diagnostics,
    variable relationships, and decision analysis. Silent if data unavailable.

    This pickle is a separate artifact from a specific attention/hallucination-
    probe pipeline, not from the current run's own explorer output, so the whole
    panel renders inside a collapsed, explicitly-labeled expander — it must never
    look like it's describing this run's own analysis."""
    if root is None or not _viz_ready() or load_case_matrix is None:
        return
    try:
        df = load_case_matrix(root)
    except Exception:
        df = None
    if df is None or df.empty:
        return
    sigs = continuous_signals(df)

    st.markdown("---")
    with st.expander(
        "📈 Attention-probe statistical analysis (from a separate m1_state.pkl, if present)",
        expanded=False,
    ):
        cov = ", ".join(f"{s}: {int(df[s].notna().sum())}" for s in sigs) if sigs else "—"
        st.caption(
            f"Reconstructed from a frozen M1 state pickle written by a specific "
            f"attention/hallucination-probe pipeline ({len(df)} cases) — a different "
            f"artifact from this run's own explorer analysis, and not guaranteed to "
            f"belong to this run if the output directory was reused. Continuous-signal "
            f"coverage — {cov}. Charts below go beyond FAIL-vs-PASS means: discrimination "
            f"(ROC/coef), error structure (confusion), distribution shape, and relationships."
        )

        # ── Model evaluation ──────────────────────────────────────────────
        st.markdown("**Model evaluation**")
        c1, c2 = st.columns(2)
        has_answers = df["model_yes"].notna().any() and df["truth_yes"].notna().any()
        if has_answers:
            with c1:
                st.plotly_chart(
                    viz.confusion_matrix(df["truth_yes"], df["model_yes"],
                                         pos_label="Yes (present)", neg_label="No (absent)",
                                         title="Model answer vs ground truth"),
                    width="stretch",
                    key="stat_confusion_matrix",
                )
                st.caption("FP = hallucination (said Yes, object absent); FN = miss.")
        if sigs:
            with (c2 if has_answers else c1):
                st.plotly_chart(viz.roc_curves(df, sigs, label_col="is_fail"),
                                width="stretch", key="stat_roc_curves")
                st.caption("How well each signal alone separates FAIL from PASS (AUC).")
            st.plotly_chart(viz.coef_plot(df, sigs, label_col="is_fail"),
                            width="stretch", key="stat_coef_plot")
            st.caption("Standardized univariate logistic coefficients (bootstrap 95% CI); "
                       "comparable across signals — CI crossing 0 ⇒ not significant.")

        # ── Distribution diagnostics (interactive signal picker) ──────────
        if sigs:
            st.markdown("**Distribution diagnostics**")
            # default to the most discriminative signal (highest |AUC-0.5|)
            aucs = {s: abs(viz._roc(df[s].to_numpy(float), df["is_fail"].to_numpy(float))[2] - 0.5)
                    for s in sigs}
            default = max(aucs, key=aucs.get)
            pick = st.selectbox("signal", sigs, index=sigs.index(default),
                                format_func=lambda s: viz.short(s), key="stat_sig")
            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(viz.violin_by_outcome(df, pick), width="stretch", key=f"stat_violin_{pick}")
                st.plotly_chart(viz.ecdf_by_outcome(df, pick), width="stretch", key=f"stat_ecdf_{pick}")
            with d2:
                st.plotly_chart(viz.kde_by_outcome(df, pick), width="stretch", key=f"stat_kde_{pick}")
                st.plotly_chart(viz.qq_normal(df, pick), width="stretch", key=f"stat_qq_{pick}")

        # ── Variable relationships ─────────────────────────────────────────
        rel_sigs = sigs + (["probe1_fd"] if "probe1_fd" in df.columns else [])
        if len(rel_sigs) >= 2:
            st.markdown("**Variable relationships**")
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(viz.corr_heatmap(df, rel_sigs), width="stretch", key="stat_corr_heatmap")
            with r2:
                if len(sigs) >= 2:
                    st.plotly_chart(viz.quadrant(df, sigs[0], sigs[1]), width="stretch", key="stat_quadrant")

        # ── Decision analysis: which prompt strategies fix vs break cases ──
        pareto_items = [
            ("describe-first fixes", "pc_fixed_describe"),
            ("sensitive fixes", "pc_fixed_sensitive"),
            ("describe-first breaks", "pc_broken_describe"),
            ("sensitive breaks", "pc_broken_sensitive"),
        ]
        avail = [(lbl, col) for lbl, col in pareto_items if col in df.columns]
        if avail:
            vals = [int(df[col].fillna(False).astype(bool).sum()) for _, col in avail]
            if sum(vals) > 0:
                st.markdown("**Decision analysis**")
                st.plotly_chart(viz.pareto([lbl for lbl, _ in avail], vals,
                                           title="Prompt-strategy repairs vs regressions"),
                                width="stretch", key="stat_prompt_pareto")
                st.caption("How many cases each reprompting strategy fixed vs broke — "
                           "ranks whether prompt-level fixes are worth pursuing.")


def _render_diagnostic_report(
    report: dict[str, Any],
    story: dict[str, Any],
    explore_report: dict[str, Any] | None,
    fallback_conclusion: str,
) -> None:
    """Claim-first semantic report compiled from raw artifacts."""
    if not report:
        _render_run_briefing(story, explore_report, fallback_conclusion)
        return

    st.markdown("### Diagnostic report")
    confidence = str(report.get("confidence") or "unknown")
    st.markdown(
        f"""
        <div class="ev-report-answer">
          <div class="ev-brief-label">Answer first · confidence: {_html_escape(confidence)}</div>
          <div class="ev-report-answer-text">{_html_escape(str(report.get('answer') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    evidence = {
        str(e.get("id")): e for e in (report.get("evidence") or [])
        if isinstance(e, dict) and e.get("id")
    }
    if claims:
        st.markdown("#### Claims and evidence")
        for claim in claims:
            _render_claim_card(claim, evidence)

    timeline = [s for s in (report.get("timeline") or []) if isinstance(s, dict)]
    if timeline:
        with st.expander("Investigation timeline", expanded=False):
            for step in timeline:
                st.markdown(
                    f"**{step.get('stage', '')}: {step.get('title', '')}**  \n"
                    f"{step.get('summary', '')}"
                )

    readings = [r for r in (report.get("chart_readings") or []) if isinstance(r, dict)]
    if readings:
        with st.expander("Chart readings written by the agent", expanded=False):
            st.dataframe(pd.DataFrame(readings), width="stretch", hide_index=True)

    critique = [str(c) for c in (report.get("critique") or [])]
    if critique:
        with st.expander("Critique and limits", expanded=True):
            for note in critique:
                st.markdown(f"- {note}")

    actions = [str(a) for a in (report.get("next_actions") or [])]
    if actions:
        with st.expander("Recommended next actions", expanded=False):
            for action in actions:
                st.markdown(f"- {action}")


def _render_claim_card(claim: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> None:
    status = str(claim.get("status") or "descriptive")
    cls = {
        "supported": "ev-claim-supported",
        "inconclusive": "ev-claim-inconclusive",
        "refuted": "ev-claim-refuted",
    }.get(status, "ev-claim-descriptive")
    st.markdown(
        f"""
        <div class="ev-claim-card {cls}">
          <div class="ev-claim-top">
            <span class="ev-pill">{_html_escape(str(claim.get('id', 'claim')))}</span>
            <span class="ev-pill">{_html_escape(status)}</span>
          </div>
          <div class="ev-claim-text">{_html_escape(str(claim.get('text') or ''))}</div>
          <div class="ev-signal-body">{_html_escape(str(claim.get('interpretation') or ''))}</div>
          <div class="ev-signal-test">Do not infer: {_html_escape(str(claim.get('do_not_infer') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = []
    for ev_id in claim.get("evidence_ids") or []:
        ev = evidence.get(str(ev_id))
        if ev:
            rows.append({
                "id": ev.get("id"),
                "kind": ev.get("kind"),
                "title": ev.get("title"),
                "summary": ev.get("summary"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    downstream = [str(x) for x in (claim.get("downstream") or [])]
    if downstream:
        st.caption("Downstream: " + " · ".join(downstream))


def _render_plain_findings(report: dict[str, Any]) -> None:
    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    if not claims:
        return
    supported = [c for c in claims if c.get("status") == "supported"]
    descriptive = [c for c in claims if c.get("status") == "descriptive"]
    inconclusive = [c for c in claims if c.get("status") == "inconclusive"]

    cols = st.columns(3)
    groups = [
        ("Main supported findings", supported, "These are the findings to carry forward."),
        ("Sanity checks / descriptive", descriptive, "Useful context, not root-cause evidence."),
        ("Not supported", inconclusive, "Tested but not convincing in this run."),
    ]
    for col, (title, rows, caption) in zip(cols, groups, strict=False):
        with col:
            st.markdown(f"**{title}**")
            if rows:
                for claim in rows[:4]:
                    st.markdown(f"- {_html_escape(str(claim.get('text') or ''))}")
            else:
                st.caption("None")
            st.caption(caption)


def _render_run_briefing(
    story: dict[str, Any],
    explore_report: dict[str, Any] | None,
    conclusion: str,
) -> None:
    """A human-first summary before the dense plots. The dashboard has many
    diagnostics; this pins the reader to what was actually done and what evidence
    is allowed to support a claim."""
    signals = _candidate_signals(explore_report)
    supported = [
        str(s.get("name"))
        for s in signals
        if s.get("reject") is True and not _is_leaky_signal(s)
    ]
    leaky = [str(s.get("name")) for s in signals if _is_leaky_signal(s)]
    diagnoses = story.get("diagnoses") or []
    surgeries = story.get("surgeries") or []
    fixes = story.get("fixes") or []
    charts = [c for c in (explore_report or {}).get("charts", []) if isinstance(c, dict)]
    split = ((explore_report or {}).get("adjudication") or {}).get("split", "unknown")

    takeaway = (
        _truncate(str(conclusion), 220)
        if conclusion else
        (
            f"Supported signals: {', '.join(supported[:3])}"
            if supported else
            "No held-out supported signal was found in the loaded report."
        )
    )
    evidence = (
        f"{len(supported)} non-leaky signal(s) survived held-out e-BH"
        if supported else
        "No non-leaky signal survived held-out e-BH"
    )
    if leaky:
        label = "sanity check" if len(leaky) == 1 else "sanity checks"
        evidence += f"; {len(leaky)} {label} demoted"
    downstream = (
        f"{sum(len(d.get('hypotheses') or []) for d in diagnoses)} hypothesis(es), "
        f"{len(surgeries)} test/intervention event(s), {len(fixes)} fix event(s)"
    )

    st.markdown("### Run briefing")
    st.markdown(
        f"""
        <div class="ev-brief-grid">
          <div class="ev-brief-card">
            <div class="ev-brief-label">Question answered</div>
            <div class="ev-brief-value">{_html_escape(takeaway)}</div>
          </div>
          <div class="ev-brief-card">
            <div class="ev-brief-label">Evidence you can trust</div>
            <div class="ev-brief-value">{_html_escape(evidence)}</div>
          </div>
          <div class="ev-brief-card">
            <div class="ev-brief-label">Pipeline stages loaded</div>
            <div class="ev-brief-value">
              Explore charts: {len(charts)} · confirm split: {_html_escape(str(split))}<br/>
              {_html_escape(downstream)}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_interpretation_guide() -> None:
    with st.expander("How to interpret this page", expanded=True):
        st.markdown(
            """
            - **Exploratory observations and charts** show patterns the agent found.
              Treat them as leads, not proof.
            - **Signal effect sizes** are the confirmatory layer. A signal matters
              only if it survives the held-out e-BH check; grey/inconclusive rows
              are descriptive.
            - **Leaky signals** re-measure the label or a near-label proxy. They can
              validate plumbing, but they should not be read as root causes.
            - **Hypotheses** are M3's explanations formed from the confirmed signals
              plus exploratory context. They are not accepted until M5/M4 tests
              support them.
            - **Fix outcomes** are the final gate. A plausible hypothesis without a
              validated fix is still only a diagnosis candidate.
            """
        )


def _render_visual_plan(explore_report: dict[str, Any] | None) -> None:
    plan = [
        p for p in ((explore_report or {}).get("visual_plan") or [])
        if isinstance(p, dict)
    ]
    if not plan:
        st.caption(
            "Visualization plan: not present in this report. Newer explorer runs "
            "record why each plot type was selected."
        )
        return
    rows = []
    for item in plan:
        cols = item.get("required_columns") or []
        if isinstance(cols, list):
            cols_text = ", ".join(str(c) for c in cols)
        else:
            cols_text = str(cols)
        rows.append({
            "visual": item.get("name", ""),
            "question": item.get("question", ""),
            "data_shape": item.get("data_shape", ""),
            "plot_kind": item.get("plot_kind", ""),
            "why this plot": item.get("rationale", ""),
            "columns": cols_text,
        })
    with st.expander("Why these charts were chosen", expanded=False):
        st.caption(
            "This is the agent's intermediate visualization plan: the chart type "
            "decision, data shape, and rationale before code was written."
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


_VERDICT_PILL = {
    "supported": ("ev-pill-ok", "Supported"),
    "refuted": ("ev-pill-fail", "Not supported"),
    "inconclusive": ("ev-pill-warn", "Inconclusive"),
}


def _render_hypothesis_card(h: dict[str, Any], *, descriptive: bool = True) -> None:
    """One M3 hypothesis with its cited evidence. When this run has an M5
    verdict for it (``h["m5"]``) and the run isn't descriptive-only, a
    verdict pill renders next to the statement — otherwise, no verdict."""
    refs = h.get("referenced_charts") or []
    ref_line = ("based on: " + ", ".join(str(r) for r in refs)) if refs else ""
    test_design = str(h.get("test_design") or "").strip()
    test_block = (
        f'<div class="ev-signal-test">How this could be checked: {_html_escape(test_design)}</div>'
        if test_design else ""
    )
    pill = ""
    m5 = h.get("m5")
    if not descriptive and m5:
        cls, label = _VERDICT_PILL.get(str(m5.get("status")), ("ev-pill", str(m5.get("status") or "untested")))
        pill = f' <span class="ev-pill {cls}">{_html_escape(label)}</span>'
    st.markdown(
        f"""
        <div class="ev-signal">
          <div class="ev-signal-title">{_html_escape(str(h.get('statement', '')))}{pill}</div>
          <div class="ev-signal-body">failure mode: <b>{_html_escape(str(h.get('failure_mode', '')))}</b>
            {('· ' + _html_escape(ref_line)) if ref_line else ''}</div>
          {test_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_loop_flow(story, explore_report, explore_dir) -> None:
    """The ordered narrative: explore notes → M2 → M3 hypotheses → (when this
    run isn't descriptive-only) M4/M5 tests and Fix outcomes."""
    if explore_report:
        with st.expander("Step 1 — exploratory observations (UNCONFIRMED; fed to M3 only)", expanded=False):
            for o in (explore_report.get("observations") or [])[:10]:
                st.markdown(f"- {o}")
            for c in (explore_report.get("caveats") or [])[:8]:
                st.caption(f"⚠ {c}")

    analyses = story.get("analyses") or []
    diagnoses = story.get("diagnoses") or []

    if analyses:
        st.markdown("### M2 — exploratory analysis")
        for a in analyses:
            sev = a.get("severity") or ""
            st.markdown(f"- **Cycle {a.get('cycle')}** {('· severity ' + str(sev)) if sev else ''}")
            concl = a.get("conclusion") or a.get("narrative") or ""
            if concl:
                st.caption(_truncate(str(concl), 320))

    st.markdown("### M3 — hypotheses")
    if not diagnoses:
        st.caption("No diagnosis events in the loop log.")
    for diag in diagnoses:
        st.markdown(f"**Cycle {diag.get('cycle')}** · {diag.get('n_hypotheses', 0)} hypotheses")
        refs = diag.get("referenced_charts") or []
        if refs:
            st.caption("M3 referenced explore artifacts: " + ", ".join(str(r) for r in refs))
        for h in diag.get("hypotheses") or []:
            test_design = str(h.get("test_design") or "").strip()
            test_line = (
                f'<div class="ev-signal-test">how to check: {_html_escape(test_design)}</div>'
                if test_design else ""
            )
            st.markdown(
                f"""
                <div class="ev-signal">
                  <div class="ev-signal-title">{_html_escape(str(h.get('statement', '')))}</div>
                  <div class="ev-signal-test">failure_mode: {_html_escape(str(h.get('failure_mode', '')))}</div>
                  {test_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
        raw = str(diag.get("raw_judge_output") or "").strip()
        if raw:
            with st.expander(f"Full M3 reasoning — cycle {diag.get('cycle')} (raw judge output)", expanded=False):
                st.text(raw)

    if not _story_is_descriptive(story):
        surgeries = story.get("surgeries") or []
        m4 = [s for s in surgeries if s.get("module") == "m4"]
        m5 = [s for s in surgeries if s.get("module") == "m5"]
        if m4 or m5:
            st.markdown("### M4/M5 — mechanism + repair tests")
            for s in m4 + m5:
                cls, label = _VERDICT_PILL.get(
                    str(s.get("status")), ("ev-pill", str(s.get("status") or "untested"))
                )
                st.markdown(
                    f"""
                    <div class="ev-signal">
                      <div class="ev-signal-title">{_html_escape(str(s.get('hypothesis', '')))}
                        <span class="ev-pill {cls}">{_html_escape(label)}</span></div>
                      <div class="ev-signal-body">{_html_escape(str(s.get('module', '')).upper())} · cycle {s.get('cycle')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        fixes = story.get("fixes") or []
        if fixes:
            st.markdown("### Fix")
            for f in fixes:
                st.caption(f"Cycle {f.get('cycle')}: fix event recorded.")


def _signal_effect_figure(signals: list[dict[str, Any]]):
    """Horizontal bar of each signal's effect size, coloured by host verdict
    (REJECT H0 = real association vs inconclusive). Returns a matplotlib Figure
    or None when matplotlib is missing / there is nothing numeric to plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = [(str(s.get("display_name") or s.get("name", "?")), str(s.get("name", "?")),
             float(s["effect"]), bool(s.get("reject")), str(s.get("source", "")))
            for s in signals if isinstance(s.get("effect"), (int, float))]
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])
    names = [f"{display_name(label or raw, compact=True)}  ({src})" for label, raw, _, _, src in rows]
    effects = [e for _, _, e, _, _ in rows]
    colors = ["#0f8a5f" if rej else "#9aa4b2" for _, _, _, rej, _ in rows]

    fig, ax = plt.subplots(figsize=(7.2, max(1.6, 0.55 * len(rows) + 0.8)))
    ax.barh(range(len(rows)), effects, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("effect size (failure association)")
    ax.set_title("Green = REJECT H0 (real, e-BH); grey = inconclusive")
    fig.tight_layout()
    return fig


def _is_leaky_signal(s: dict[str, Any]) -> bool:
    """Target leakage = the outcome re-measured. Signature: perfect separation
    (effect ~1.0 with a zero-width CI), or a signal derived from the probe that
    defines the failure label. Such a signal must never be ranked #1 (skill §4)."""
    name = str(s.get("name", "")).lower()
    if name.startswith("probe1") or "false_detection" in name:
        return True
    eff = s.get("effect")
    ci = s.get("ci") or [None, None]
    try:
        lo, hi = float(ci[0]), float(ci[1])
        if eff is not None and abs(float(eff)) >= 0.999 and (hi - lo) <= 1e-6:
            return True
    except (TypeError, ValueError, IndexError):
        pass
    return False


def _forest_rows(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt host signal dicts to eval_viz_theme.forest_effects row schema."""
    rows = []
    for s in signals:
        if not isinstance(s.get("effect"), (int, float)):
            continue
        ci = s.get("ci") or [None, None]
        raw = str(s.get("name", "?"))
        rows.append({
            "signal": str(s.get("display_name") or display_name(raw)),
            "raw_signal": raw,
            "effect": float(s["effect"]),
            "ci_lo": ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
            "ci_hi": ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
            "significant": bool(s.get("reject")),
            "leaky": _is_leaky_signal(s),
        })
    return rows


def _render_confirmatory_evidence(
    signals: list[dict[str, Any]],
    explore_report: dict[str, Any] | None = None,
    explore_dir: Path | None = None,
    *,
    descriptive: bool = False,
) -> None:
    rows = _evidence_rows(signals, descriptive=descriptive)
    diagnostic = [r for r in rows if r["role"] == "diagnostic signal"]
    supported = [r for r in rows if r["decision"] == "Supported" and r["role"] == "diagnostic signal"]
    audits = [r for r in rows if r["role"] == "sanity check"]
    lead = supported[0] if supported else None

    if descriptive:
        lead_desc = diagnostic[0] if diagnostic else None
        summary = (
            f"{lead_desc['finding']} shows the largest FAIL-vs-PASS separation in this run "
            f"(effect {lead_desc['effect']}, CI {lead_desc['ci']}) — shown descriptively, "
            "not yet adjudicated. Run the confirm phase to test whether it holds out."
            if lead_desc else
            "Candidate signals are listed descriptively; run the confirm phase to adjudicate them."
        )
    elif lead:
        summary = (
            f"{lead['finding']} is the main usable finding. FAIL cases have higher values "
            f"for this signal on the confirmation split (effect {lead['effect']}, CI {lead['ci']})."
        )
    elif audits:
        summary = (
            "Only sanity-check signals were supported. That validates the plumbing, "
            "but it does not explain the model failure."
        )
    else:
        summary = "No non-leaky signal was supported on the confirmation split."

    st.markdown(
        f"""
        <div class="ev-evidence-summary">
          <div class="ev-brief-label">How to read this evidence</div>
          <div class="ev-storyboard-summary">{_html_escape(summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    signal_by_name = {str(s.get("name") or ""): s for s in signals}
    for row in rows:
        signal = signal_by_name.get(row["raw_signal"], {})
        _render_evidence_panel(row, signal, explore_report or {}, explore_dir, descriptive=descriptive)
    st.caption(
        "Cards are computed by the host. Positive effect means the signal is more "
        "common/larger in FAIL than PASS; CI crossing 0 means weak separation. "
        + ("Verdicts are deferred to the confirm phase; these are descriptive only. "
           if descriptive else
           "Sanity checks validate the pipeline and are not root-cause explanations.")
    )

    with st.expander("Technical forest plot", expanded=False):
        if _viz_ready():
            st.plotly_chart(viz.forest_effects(_forest_rows(signals)), width="stretch", key="analysis_forest_effects")
        else:
            fig = _signal_effect_figure(signals)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
        st.dataframe(_signals_dataframe(signals), width="stretch", hide_index=True)


def _render_evidence_panel(
    row: dict[str, str],
    signal: dict[str, Any],
    explore_report: dict[str, Any],
    explore_dir: Path | None,
    *,
    descriptive: bool = False,
) -> None:
    with st.container(border=True):
        kind = "ev-evidence-audit" if row["role"] == "sanity check" else "ev-evidence-signal"
        st.markdown(
            f"""
            <div class="ev-evidence-card {kind}">
              <div class="ev-evidence-card-top">
                <span class="ev-evidence-name">{_html_escape(row['finding'])}</span>
                <span class="ev-evidence-pill">{_html_escape(row['decision'])}</span>
              </div>
              <div class="ev-evidence-meaning">{_html_escape(row['plain meaning'])}</div>
              <div class="ev-evidence-meta">
                <span>{_html_escape(row['role'])}</span>
                <span>effect {_html_escape(row['effect'])}</span>
                <span>CI {_html_escape(row['ci'])}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        charts = _supporting_charts_for_signal(signal, explore_report)
        reading = _supporting_reading_for_signal(signal, explore_report, charts)
        takeaway = _evidence_takeaway(row, signal, descriptive=descriptive)
        st.markdown(
            f"""
            <div class="ev-evidence-takeaway">
              <div class="ev-brief-label">Takeaway</div>
              <div>{_html_escape(takeaway)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Supporting experiment**")
        if charts and explore_dir is not None:
            if reading:
                st.caption(reading)
            # Two-up, matching every other chart surface — a supporting figure
            # rendered full-bleed here would tower over the identical figure
            # shown half-width one tab away.
            ev_cols = st.columns(2)
            for idx, chart in enumerate(charts[:2]):
                with ev_cols[idx % 2]:
                    _render_chart_card(
                        chart,
                        explore_dir,
                        heading_level="caption",
                        prefer_rendered_artifact=False,
                        key_prefix=f"evidence_{row['raw_signal']}_{idx}",
                    )
        else:
            st.caption("Held-out effect estimate and confidence interval above.")


def _evidence_rows(signals: list[dict[str, Any]], *, descriptive: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in signals:
        leaky = _is_leaky_signal(s)
        label = display_name(s.get("name") if leaky else (s.get("display_name") or s.get("name")))
        reject = s.get("reject")
        effect = _fmt_effect(s.get("effect"))
        ci = _fmt_ci(s.get("ci"))
        if leaky:
            role = "sanity check"
            if not label.lower().startswith("sanity check"):
                label = f"Sanity check: {label[:1].lower()}{label[1:]}"
            meaning = (
                "This mostly mirrors the known FAIL/PASS label. It confirms the measurement "
                "pipeline, but it should not be treated as an explanation."
            )
        else:
            role = "diagnostic signal"
            if descriptive:
                meaning = (
                    f"{label}: FAIL/PASS group means differ by effect {effect} in this run "
                    "(descriptive; validity not yet tested)."
                )
            else:
                meaning = (
                    f"{label} separates FAIL from PASS in this run."
                    if reject is True else
                    f"{label} was tested but did not clearly separate FAIL from PASS."
                )
        if descriptive:
            # Analysis phase: no verdict — the explorer's reject flag is not confirmation.
            decision = "Descriptive"
        else:
            decision = "Supported" if reject is True else ("Not supported" if reject is False else "Descriptive")
        rows.append({
            "finding": label,
            "raw_signal": str(s.get("name") or ""),
            "role": role,
            "decision": decision,
            "effect": effect,
            "ci": ci,
            "plain meaning": meaning,
        })
    # In descriptive mode rank diagnostics by |effect| (no verdict to sort on).
    if descriptive:
        def _abs_eff(s: dict[str, Any]) -> float:
            try:
                return abs(float(s.get("effect")))
            except (TypeError, ValueError):
                return 0.0
        eff_by_name = {str(s.get("name") or ""): _abs_eff(s) for s in signals}
        rows.sort(key=lambda r: (r["role"] != "diagnostic signal", -eff_by_name.get(r["raw_signal"], 0.0), r["finding"]))
    else:
        rows.sort(key=lambda r: (r["role"] != "diagnostic signal", r["decision"] != "Supported", r["finding"]))
    return rows


def _supporting_charts_for_signal(signal: dict[str, Any], explore_report: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = _signal_tokens(signal)
    if not tokens:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, chart in enumerate(explore_report.get("charts") or []):
        if not isinstance(chart, dict):
            continue
        blob = " ".join(str(chart.get(k, "")) for k in ("name", "title", "display_name", "data", "x", "y"))
        score = sum(1 for token in tokens if token and token in blob)
        if score:
            priority = 0 if str(chart.get("name", "")).startswith("failrate_by_") else 1
            scored.append((score * 10 - priority, idx, chart))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chart for _, _, chart in scored]


def _supporting_reading_for_signal(
    signal: dict[str, Any],
    explore_report: dict[str, Any],
    charts: list[dict[str, Any]],
) -> str:
    chart_blobs = [
        " ".join(str(c.get(k) or "").lower() for k in ("name", "title", "display_name"))
        for c in charts
    ]
    tokens = _signal_tokens(signal)
    for reading in explore_report.get("chart_readings") or []:
        if not isinstance(reading, dict):
            continue
        chart = str(reading.get("chart") or "").lower()
        text = str(reading.get("reading") or "").strip()
        if not text:
            continue
        if any(token in chart or token in text for token in tokens):
            return text
        if any(chart and chart in blob for blob in chart_blobs):
            return text
    return "The chart below is the experiment behind this finding: it shows the same signal split by FAIL/PASS outcome."


def _evidence_takeaway(row: dict[str, str], signal: dict[str, Any], *, descriptive: bool = False) -> str:
    name = row["finding"]
    if row["role"] == "sanity check":
        return f"{name} validates the measurement path, but it is not a root-cause explanation."
    if descriptive:
        return (
            f"{name} is a descriptive lead (effect {row['effect']}, CI {row['ci']}). "
            "Its validity is deferred to the confirm phase — do not treat it as confirmed yet."
        )
    if signal.get("reject") is True:
        return (
            f"{name} is the actionable M2 finding: it separates FAIL from PASS on the "
            f"confirmation split with effect {row['effect']} and CI {row['ci']}."
        )
    if signal.get("reject") is False:
        return (
            f"{name} was tested, but the confirmation split did not support it as a "
            "reliable FAIL/PASS separator."
        )
    return f"{name} is descriptive context only; it needs a held-out confirmation before use."


def _signal_tokens(signal: dict[str, Any]) -> list[str]:
    import re

    tokens: list[str] = []
    for value in (signal.get("name"), signal.get("display_name")):
        text = str(value or "").strip()
        if text:
            tokens.append(text)
    recipe = signal.get("recipe")
    if isinstance(recipe, dict):
        expr = str(recipe.get("expr") or "")
        tokens.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen and token not in {"and", "or", "not"}:
            seen.add(token)
            out.append(token)
    return out



def _fmt_effect(value: Any) -> str:
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ci(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"{float(value[0]):+.2f} to {float(value[1]):+.2f}"
        except (TypeError, ValueError):
            return "—"
    return "—"


def _signals_dataframe(signals: list[dict[str, Any]]):
    def _verdict(s):
        if _is_leaky_signal(s):
            return "leaky (re-measured label)"
        if s.get("reject") is True:
            return "REJECT H0"
        if s.get("reject") is False:
            return "inconclusive"
        return "descriptive"

    def _fmt(x, kind):
        return viz.fmt(x, kind) if viz is not None else x

    rows = [{
        "signal": display_name(s.get("display_name") or s.get("name"), compact=True),
        "raw field": s.get("name"),
        "source": s.get("source"),
        "effect": _fmt(s.get("effect"), "effect"),
        "e_value": _fmt(s.get("e_value"), "stat") if s.get("e_value") is not None else "—",
        "verdict": _verdict(s),
        "recipe": (s.get("recipe") or {}).get("expr") if isinstance(s.get("recipe"), dict) else None,
    } for s in signals]
    return pd.DataFrame(rows)


def _explore_table_dirs(explore_dir: Path) -> list[Path]:
    return [d for d in (explore_dir / "tables", explore_dir / "sandbox" / "tables") if d.is_dir()]


def _scatter_axis_names(explore_report, csv_name: str) -> tuple[str | None, str | None]:
    """Recover (x_signal, y_signal) for a scatter CSV from the report chart whose
    title is '<xsig> vs <ysig> by outcome'. Returns (None, None) if not found."""
    stem = csv_name.replace(".csv", "")
    for c in (explore_report or {}).get("charts", []):
        if not isinstance(c, dict):
            continue
        if c.get("name") == stem or str(c.get("data", "")).endswith(csv_name):
            title = str(c.get("title", ""))
            if " vs " in title:
                left, _, right = title.partition(" vs ")
                right = right.split(" by outcome")[0].strip()
                return left.strip() or None, right or None
    return None, None


def _resolve_scatter_axes(d, explore_report, name):
    """Resolve a scatter CSV's (x_sig, y_sig, outcome_col), tolerating both schemas.

    Newer fused tables store the real signal names as columns directly
    (``[attention_entropy, center_offset, outcome]``); older ones used literal
    ``x``/``y`` recovered from the report's chart title. Falls back to the CSV's
    own non-outcome columns so a non-existent name is never handed to plotly.
    Returns ``(df, x_sig, y_sig, outcome_col)`` with the trailing three None when
    a two-axis scatter cannot be formed."""
    outcome_col = next((c for c in d.columns if c.lower() == "outcome"), None)
    xs, ys = _scatter_axis_names(explore_report, name)
    if {"x", "y"} <= set(d.columns) and xs and ys:
        d = d.rename(columns={"x": xs, "y": ys})  # legacy x/y → recovered names
    if not (xs and ys and xs in d.columns and ys in d.columns):
        value_cols = [c for c in d.columns if c != outcome_col]
        xs, ys = (value_cols + [None, None])[:2]
    if not (xs and ys and xs in d.columns and ys in d.columns and outcome_col):
        return d, None, None, None
    return d, xs, ys, outcome_col


def _counts_bar_agg(df: pd.DataFrame):
    """Class balance from an aggregated (outcome, count) table → a single slim
    100%-stacked composition strip (not two fat bars for two numbers)."""
    rowmap = {str(r["outcome"]).upper(): int(r["count"]) for _, r in df.iterrows()}
    order = [g for g in ("FAIL", "PASS") if g in rowmap] or list(rowmap)
    return viz.composition_bar(order, [rowmap[g] for g in order])


def _groupstats_dumbbell(df: pd.DataFrame, signal: str):
    """FAIL vs PASS group means as a dumbbell (not a two-bar). Only mean/median are
    persisted, so spread cannot be shown — labelled honestly as means."""
    import plotly.graph_objects as go
    rowmap = {str(r["outcome"]).upper(): r for _, r in df.iterrows()}
    fig = go.Figure()
    present = [g for g in ("FAIL", "PASS") if g in rowmap]
    if {"FAIL", "PASS"} <= set(rowmap):
        fig.add_trace(go.Scatter(
            x=[rowmap["FAIL"]["mean"], rowmap["PASS"]["mean"]], y=[signal, signal],
            mode="lines", line=dict(color=viz.PALETTE["AXIS"], width=2),
            showlegend=False, hoverinfo="skip"))
    for g in present:
        m = rowmap[g]
        fig.add_trace(go.Scatter(
            x=[m["mean"]], y=[signal], mode="markers+text",
            marker=dict(color=viz.outcome_color(g), size=14),
            text=[f"{g} {viz.fmt(m['mean'], 'val')}"],
            textposition="top center", textfont=dict(size=11),
            name=g, hovertext=f"{g}: mean={viz.fmt(m['mean'],'val')} median={viz.fmt(m.get('median'),'val')}",
            hoverinfo="text"))
    fig.update_layout(title=f"{viz.short(signal)} — group means (FAIL vs PASS)",
                      xaxis_title=viz.short(signal), yaxis_title="", showlegend=False,
                      height=180, yaxis=dict(showticklabels=False))
    return fig


def _failrate_scatter(df: pd.DataFrame, signal: str):
    """Fail rate vs a binned signal: markers at bin midpoints, size ∝ n, with
    human-readable bin labels — replaces the bar/line over machine bin edges."""
    import re

    import plotly.graph_objects as go
    bincol = df.columns[0]
    labels, mids, rates, ns = [], [], [], []
    for _, r in df.iterrows():
        nums = re.findall(r"-?\d+\.?\d*", str(r[bincol]))
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
            labels.append(viz.human_bins([lo, hi])[0])
            mids.append((lo + hi) / 2)
        else:
            labels.append(str(r[bincol]))
            mids.append(len(mids))
        rates.append(float(r.get("fail_rate", 0)))
        ns.append(int(r.get("n", 0)) if pd.notna(r.get("n")) else 0)
    sizes = [8 + 28 * (n / max(ns)) for n in ns] if ns and max(ns) else [10] * len(ns)
    fig = go.Figure(go.Scatter(
        x=mids, y=rates, mode="lines+markers+text",
        line=dict(color=viz.PALETTE["FAIL"], width=2),
        marker=dict(color=viz.PALETTE["FAIL"], size=sizes,
                    line=dict(color="white", width=1)),
        text=[f"n={n}" for n in ns], textposition="top center",
        textfont=dict(size=10), hovertext=labels, hoverinfo="text+y"))
    fig.update_layout(title=f"Fail rate vs {viz.short(signal)} (by bin; marker size ∝ n)",
                      xaxis_title=viz.short(signal), yaxis_title="fail rate",
                      yaxis=dict(range=[-0.05, 1.05]), height=320)
    return fig


def _render_explore_tables(explore_report, explore_dir, *, full: bool, root=None) -> None:
    """Surface the CSV tables the explorer wrote. In the Analysis tab (full=False)
    each table is shown as the chart its content calls for (counts→bar, two-signal
    →scatter, fail-rate→binned curve, group stats→dumbbell), per the eval-chart-
    style policy. In the Tables tab (full=True) they are sortable dataframes only —
    one home per fact, no re-plotting."""
    explore_dir = Path(explore_dir)
    csvs: list[Path] = []
    for tdir in _explore_table_dirs(explore_dir):
        csvs.extend(sorted(tdir.glob("*.csv")))
    seen: set[str] = set()
    uniq = [p for p in csvs if not (p.name in seen or seen.add(p.name))]

    if not uniq:
        if full:
            st.caption("No CSV tables found for this run.")
        return

    # Tables tab: real sortable tables, no charts (don't re-plot Analysis findings).
    if full:
        st.markdown("#### Data tables")
        for path in uniq:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            st.markdown(f"**{path.name}**")
            st.dataframe(df, width="stretch", height=240)
        return

    # Analysis tab: prefer ONE combined chart per family over per-signal small-
    # multiples. When the per-case matrix is available, all signals' group means
    # and fail-rate curves collapse into one chart each (skill §2).
    if not _viz_ready():
        return _render_explore_tables_legacy(uniq)

    st.markdown("#### Data behind the analysis")

    # This run's own signals — always the primary view. NOTE: a case matrix
    # reconstructed from m1_state.pkl (below) is a SEPARATE artifact from a
    # specific attention/hallucination-probe pipeline; it is not guaranteed to
    # belong to this run (a reused output dir can carry a stale pickle from an
    # unrelated experiment), so it must never silently replace what this run's
    # own explorer tables actually say.
    matrix = None
    if root is not None and load_case_matrix is not None:
        try:
            matrix = load_case_matrix(root)
        except Exception:
            matrix = None
    sigs = continuous_signals(matrix) if (matrix is not None and continuous_signals) else []

    # Class balance — compact, unique (kept as-is).
    for path in uniq:
        if path.name.startswith("class_balance"):
            try:
                d = pd.read_csv(path)
                if {"outcome", "count"} <= {c.lower() for c in d.columns}:
                    st.plotly_chart(_counts_bar_agg(d), width="stretch", key=f"explore_counts_{path.name}")
            except Exception:
                pass
            break

    # Remaining CSVs: per-signal group-stats / fail-rate charts, plus the
    # genuinely tabular summaries (correlations, discriminators). This is
    # this run's own data — always shown, never gated on the case matrix.
    folded = ("groupstats", "failrate", "class_balance")
    for path in uniq:
        name = path.name
        if name.startswith(folded):
            continue
        if name.startswith("scatter"):
            # joint scatter with a binary axis just stacks points on two lines and
            # overlaps the marginals — suppress (the quadrant/violin views cover it).
            try:
                d = pd.read_csv(path)
            except Exception:
                continue
            d, xs, ys, outcome_col = _resolve_scatter_axes(d, explore_report, name)
            if not (xs and ys and outcome_col):
                st.caption(f"↳ scatter {name}: could not resolve its two value axes — skipped.")
                continue
            binary_axis = any(d[c].nunique(dropna=True) <= 2 for c in (xs, ys))
            if binary_axis:
                st.caption(f"↳ scatter {viz.short(xs)} vs {viz.short(ys)}: one axis is binary — "
                           "suppressed (see the quadrant / distribution views above).")
            else:
                st.plotly_chart(
                    viz.joint_scatter(d, xs, ys, outcome=outcome_col),
                    width="stretch",
                    key=f"explore_scatter_{name}",
                )
            continue
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        st.markdown(f"**{name}**")
        st.dataframe(d, width="stretch", height=200)

    # Per-signal group-stats / fail-rate panels, built from THIS RUN's own CSVs —
    # always shown, regardless of whether a case matrix also happens to exist.
    for path in uniq:
        name = path.name
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower() for c in d.columns}
        if name.startswith("groupstats") and {"outcome", "mean"} <= cols:
            st.plotly_chart(
                _groupstats_dumbbell(d, name.replace("groupstats_", "").replace(".csv", "")),
                width="stretch",
                key=f"explore_groupstats_{name}",
            )
        elif name.startswith("failrate") and "fail_rate" in cols and len(d) > 1:
            st.plotly_chart(
                _failrate_scatter(d, name.replace("failrate_by_", "").replace(".csv", "")),
                width="stretch",
                key=f"explore_failrate_{name}",
            )

    # Case-matrix charts (m1_state.pkl) are a SEPARATE, optional artifact from a
    # specific attention/hallucination-probe pipeline — opt-in and clearly
    # labeled so they are never mistaken for this run's own analysis above.
    if matrix is not None and len(sigs) >= 1:
        with st.expander(
            "Attention-probe case-matrix charts (from a separate m1_state.pkl, if present)",
            expanded=False,
        ):
            st.caption(
                "Reconstructed from a frozen M1 state pickle written by a specific "
                "attention/hallucination-probe pipeline — a different artifact from "
                "this run's own explorer tables above. If this output directory was "
                "reused across experiments, this may be stale or unrelated to the "
                "run shown elsewhere on this page."
            )
            st.plotly_chart(
                viz.groupstats_strip(matrix, sigs), width="stretch", key="explore_groupstats_combined"
            )
            st.caption("All signals' FAIL-vs-PASS group means on one standardized axis "
                       "(one row per signal — replaces a panel per signal).")
            st.plotly_chart(
                viz.failrate_percentile(matrix, sigs), width="stretch", key="explore_failrate_combined"
            )
            st.caption("Each signal's fail-rate curve over its own percentile range, overlaid "
                       "on one axis.")


def _render_explore_tables_legacy(uniq: list[Path]) -> None:
    """Fallback when plotly/viz is unavailable: the original bar-chart rendering."""
    st.markdown("#### Data tables")
    for path in uniq:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        st.markdown(f"**{path.name}**")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(df.columns) >= 2 and num_cols:
            x = df.columns[0]
            y = num_cols[0] if num_cols[0] != x else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            try:
                st.bar_chart(df, x=x, y=y, height=260)
            except Exception:
                st.dataframe(df, width="stretch", height=240)


def _count_records(root: Path) -> int | None:
    """Row count from the bundle's records.json when the profile omits it."""
    try:
        data = json.loads((root / "records.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return len(data) if isinstance(data, list) else None


def _verdict_metric(report: dict[str, Any]) -> tuple[str, Any, str]:
    """Lead tile: what the run actually concluded.

    This is the number the whole pipeline exists to produce, so it goes first.
    An explore-only run legitimately confirms nothing — that renders as
    "0/N confirmed", not as a missing tile.
    """
    adj = report.get("adjudication") or {}
    n_cand = adj.get("n_candidates")
    if not isinstance(n_cand, int) or n_cand <= 0:
        return ("Confirmed", None, "no candidates adjudicated")
    n_rej = adj.get("n_rejected") or 0
    method = str(adj.get("method") or "—")
    split = str(adj.get("split") or "")
    alpha = adj.get("alpha")
    caption = f"{method} · α={alpha}" if alpha is not None else method
    if split:
        caption += f" · {split.replace('_', '-')}"
    return ("Confirmed", f"{n_rej}/{n_cand}", caption)


def _verdict_sentence(report: dict[str, Any]) -> tuple[str, str]:
    """The run's outcome as a sentence, plus a tone class.

    Answering "so what happened?" in words belongs above the evidence, not
    inside a tile the reader has to decode. Confirming nothing is a real,
    reportable outcome here — it gets a plain statement, not a warning.
    """
    adj = report.get("adjudication") or {}
    n_cand = adj.get("n_candidates")
    if not isinstance(n_cand, int) or n_cand <= 0:
        return ("No candidate signals were adjudicated in this run.", "ev-verdict-neutral")
    n_rej = adj.get("n_rejected") or 0
    split = str(adj.get("split") or "")
    noun = "signal" if n_cand == 1 else "signals"
    held_out = split == "holdout"
    if n_rej == 0:
        return (
            f"Nothing confirmed — 0 of {n_cand} candidate {noun} cleared adjudication.",
            "ev-verdict-null",
        )
    where = "on held-out rows" if held_out else "in-sample only"
    return (
        f"{n_rej} of {n_cand} candidate {noun} confirmed, {where}.",
        "ev-verdict-ok" if held_out else "ev-verdict-partial",
    )


def _short_report_title(question: str) -> str:
    q = " ".join(str(question or "").split()).strip()
    if not q:
        return "Exploratory analysis"
    lower = q.lower()
    if "hallucination" in lower and ("fail" in lower or "failure" in lower):
        return "Hallucination failure patterns"
    if lower.startswith("what predicts "):
        target = q[len("What predicts "):].split("?", 1)[0].split(" in ", 1)[0].strip()
        if target:
            return f"What predicts {target}?"
    if lower.startswith("what drives "):
        target = q[len("What drives "):].split("?", 1)[0].split(" in ", 1)[0].strip()
        if target:
            return f"What drives {target}?"
    return _truncate(q, 72)


def _analysis_stage_label(
    confirm: dict[str, Any] | None,
    fix_report: dict[str, Any] | None,
) -> str:
    if fix_report:
        return "Exploratory analysis with validation and fix results"
    if confirm:
        return "Exploratory analysis with validation results"
    return "Exploratory analysis"


def _overview_scope_sentence(
    report: dict[str, Any],
    artifact_dir: Path,
    *,
    confirm: dict[str, Any] | None,
    fix_report: dict[str, Any] | None,
) -> str:
    profile = report.get("data_profile") or {}
    parts: list[str] = []

    rows = profile.get("loaded_rows", profile.get("n_rows"))
    if rows is None:
        rows = _count_records(artifact_dir)
    if rows is not None:
        parts.append(f"{_format_int(rows)} records")

    signals = report.get("candidate_signals") if "candidate_signals" in report else None
    if isinstance(signals, list) and signals:
        parts.append(f"{_format_int(len(signals))} candidate patterns")

    charts = report.get("charts") if "charts" in report else None
    plots = report.get("plots") if "plots" in report else None
    n_visuals = len(charts or []) + len(plots or []) if charts is not None or plots is not None else 0
    if n_visuals:
        parts.append(f"{_format_int(n_visuals)} visuals")

    if confirm:
        parts.append("validation attached")
    if fix_report:
        parts.append("fix results attached")

    return " · ".join(parts) if parts else "Report sections below contain the run details."


def _overview_metrics(
    report: dict[str, Any],
    artifact_dir: Path,
    *,
    confirm: dict[str, Any] | None,
    fix_report: dict[str, Any] | None,
) -> list[tuple[str, str, str]]:
    profile = report.get("data_profile") or {}
    observations = report.get("observations") if "observations" in report else None
    signals = report.get("candidate_signals") if "candidate_signals" in report else None
    charts = report.get("charts") if "charts" in report else None
    plots = report.get("plots") if "plots" in report else None
    hypotheses = report.get("hypotheses") if "hypotheses" in report else None

    rows = profile.get("loaded_rows", profile.get("n_rows"))
    if rows is None:
        rows = _count_records(artifact_dir)

    raw_metrics: list[tuple[str, Any, str]] = [
        _verdict_metric(report),
        ("Records", rows, "cases analysed"),
        (
            "Candidate patterns",
            len(signals) if isinstance(signals, list) else None,
            "possible follow-ups",
        ),
        (
            "Hypotheses",
            len(hypotheses) if isinstance(hypotheses, list) and hypotheses else None,
            "proposed, not validated",
        ),
        (
            "Visuals",
            len(charts or []) + len(plots or []) if charts is not None or plots is not None else None,
            "analysis charts",
        ),
        ("Findings", len(observations) if isinstance(observations, list) else None, "takeaways"),
        ("Validation", "Yes" if confirm else None, "held-out results"),
        ("Fix", "Yes" if fix_report else None, "repair results"),
    ]
    return [(label, _format_int(value), caption) for label, value, caption in raw_metrics if value is not None]


def _render_overview_metrics(metrics: list[tuple[str, str, str]]) -> None:
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for idx, (col, (label, value, caption)) in enumerate(zip(cols, metrics, strict=False)):
        lead = " ev-metric-lead" if idx == 0 and label == "Confirmed" else ""
        with col:
            st.markdown(
                f"""
                <div class="ev-metric-card{lead}">
                  <div class="ev-metric-label">{_html_escape(label)}</div>
                  <div class="ev-metric-value">{_html_escape(value)}</div>
                  <div class="ev-metric-caption">{_html_escape(caption)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_report_overview(
    root: Path,
    turn: dict[str, Any],
    report: dict[str, Any],
    artifact_dir: Path,
    *,
    confirm: dict[str, Any] | None,
    fix_report: dict[str, Any] | None,
) -> None:
    ok = bool(report.get("ok"))
    status = "finished" if ok else "failed"
    status_class = "ev-pill-ok" if ok else "ev-pill-fail"

    question = str(report.get("question") or "Exploratory analysis")
    plain_question = str(report.get("plain_question") or "").strip()
    title = plain_question or _short_report_title(question)
    has_technical = bool(
        plain_question and question.strip() and question.strip() != plain_question
    )
    verdict_text, verdict_class = _verdict_sentence(report)
    stage = _analysis_stage_label(confirm, fix_report)
    scope = _overview_scope_sentence(report, artifact_dir, confirm=confirm, fix_report=fix_report)

    st.markdown(
        f"""
        <div class="ev-header ev-report-overview">
          <div>
            <div class="ev-kicker">EvalVitals Analysis Report</div>
            <h1>{_html_escape(title)}</h1>
            <div class="ev-overview-stage">{_html_escape(stage)}</div>
            <div class="ev-verdict-line {verdict_class}">{_html_escape(verdict_text)}</div>
            <div class="ev-path">{_html_escape(scope)}</div>
          </div>
          <div class="ev-header-right">
            <span class="ev-pill {status_class}">{status}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_overview_metrics(
        _overview_metrics(report, artifact_dir, confirm=confirm, fix_report=fix_report)
    )

    with st.expander("View full research question and run details", expanded=False):
        st.markdown("**Full research question**")
        st.markdown(_html_escape(question))
        if has_technical:
            st.markdown("**Reader-friendly restatement**")
            st.markdown(_html_escape(plain_question))
        st.markdown("**Run directory**")
        st.code(str(root), language="text")
        st.caption(f"Bundle: `{root}`")
        st.markdown("**Report details**")
        st.json({
            "status": status,
            "report": turn["name"],
            "report_type": "exploratory_report.json / fused_report.json",
            "analysis_stage": stage,
            "validation_results": bool(confirm),
            "fix_results": bool(fix_report),
            "attempts": report.get("attempts"),
        })

    if report.get("error"):
        st.error(report["error"])

def _chart_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a chart's ``name`` (falling back to ``title``) -> its spec dict."""
    out: dict[str, dict[str, Any]] = {}
    for c in report.get("charts", []) or []:
        if not isinstance(c, dict):
            continue
        key = str(c.get("name") or c.get("title") or "")
        if key:
            out[key] = c
    return out


def _plot_lookup(report: dict[str, Any]) -> dict[str, str]:
    """Map an agent-authored plot's filename stem -> its path, so a takeaway
    can reference a PNG under ``figures/`` by its short name."""
    return {Path(str(p)).stem: str(p) for p in (report.get("plots") or [])}


def _render_chart_grid(
    charts: list[dict[str, Any]],
    turn_dir: Path,
    *,
    key_prefix: str,
    columns: int = 2,
    report: dict[str, Any] | None = None,
) -> None:
    if not charts:
        return
    # One st.columns() per ROW, not one for the whole grid. A single set of
    # columns makes each column an independent stack, so a tall figure in the
    # left column pushes everything below it out of step with the right — the
    # masonry effect that makes a long gallery unreadable. Fresh columns per
    # pair keeps rows aligned.
    for start in range(0, len(charts), columns):
        row = charts[start:start + columns]
        cols = st.columns(columns)
        for offset, chart in enumerate(row):
            with cols[offset]:
                idx = start + offset
                _render_chart_card(chart, turn_dir, key_prefix=f"{key_prefix}{idx}")
                if report is not None:
                    _render_visual_explanation(
                        report, name=str(chart.get("name") or chart.get("title") or ""), chart=chart
                    )


def _render_charts_and_plots(report: dict[str, Any], turn_dir: Path) -> None:
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]
    plots = report.get("plots") or []

    if not charts and not plots:
        st.caption("No charts or plots were reported.")
        return

    if charts:
        st.markdown("### Interactive Charts")
        _render_chart_grid(charts, turn_dir, key_prefix="chart_and_plot", report=report)

    if plots:
        st.markdown("### Generated Figures")
        for start in range(0, len(plots), 2):
            plot_cols = st.columns(2)
            for offset, item in enumerate(plots[start:start + 2]):
                with plot_cols[offset]:
                    _render_plot_card(item, turn_dir)
                    _render_visual_explanation(report, name=Path(str(item)).stem)


def _render_chart_card(
    chart: dict[str, Any],
    turn_dir: Path,
    *,
    heading_level: str = "title",
    prefer_rendered_artifact: bool = True,
    key_prefix: str = "chart",
) -> None:
    title = str(chart.get("display_name") or chart.get("title") or chart.get("name") or "Chart")
    title = display_name(title)
    df = _table_to_dataframe(chart.get("data"), turn_dir)

    cls = "ev-card-title" if heading_level == "title" else "ev-card-subtitle"
    st.markdown(f'<div class="{cls}">{_html_escape(title)}</div>', unsafe_allow_html=True)

    # Prefer the host-rendered PNG (deterministic, from the chart spec) when present.
    fig_path = chart.get("figure_path")
    if prefer_rendered_artifact and fig_path:
        p = _resolve_artifact_path(fig_path, turn_dir)
        if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(p), width="stretch")
            return
    if chart.get("render_skipped"):
        st.caption(f"(render skipped: {chart['render_skipped']})")

    if df is None:
        st.json(chart)
        return

    if not prefer_rendered_artifact and _viz_ready():
        signal = _signal_from_chart(chart)
        try:
            if signal and str(chart.get("name", "")).startswith("failrate_by_"):
                st.plotly_chart(
                    _failrate_scatter(df, signal),
                    width="stretch",
                    key=f"{key_prefix}_failrate_{chart.get('name', signal)}",
                )
                return
            if signal and str(chart.get("name", "")).startswith("groupstats_"):
                st.plotly_chart(
                    _groupstats_dumbbell(df, signal),
                    width="stretch",
                    key=f"{key_prefix}_groupstats_{chart.get('name', signal)}",
                )
                return
        except Exception:
            pass

    x = chart.get("x")
    y = chart.get("y")
    if x in df.columns and y in df.columns:
        kind = str(chart.get("kind", "bar")).lower()
        if kind in {"line", "timeseries"}:
            st.line_chart(df, x=x, y=y, height=280)
        else:
            st.bar_chart(df, x=x, y=y, height=280)
    else:
        st.dataframe(df, width="stretch", height=280)


def _signal_from_chart(chart: dict[str, Any]) -> str | None:
    for value in (chart.get("name"), chart.get("data"), chart.get("x"), chart.get("title")):
        text = str(value or "").rsplit("/", 1)[-1].removesuffix(".csv")
        for prefix in ("failrate_by_", "groupstats_"):
            if text.startswith(prefix):
                return text[len(prefix):]
        if text.endswith("_bin"):
            return text.removesuffix("_bin")
    return None


def _render_plot_card(item: Any, turn_dir: Path) -> None:
    path = _resolve_artifact_path(item, turn_dir)
    title = display_name(Path(str(item)).stem)
    st.markdown(f'<div class="ev-card-title">{_html_escape(title)}</div>', unsafe_allow_html=True)
    if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        st.image(str(path), width="stretch")
    else:
        st.caption(f"Missing plot: {item}")


def _render_tables(report: dict[str, Any], turn_dir: Path) -> None:
    tables = report.get("tables") or {}
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]
    chart_tables = {
        str(c.get("name") or c.get("title") or f"chart_{idx}"): c.get("data")
        for idx, c in enumerate(charts)
        if c.get("data") is not None
    }

    if not tables and not chart_tables:
        st.caption("No structured tables were reported.")
        return

    names = list(tables.keys()) + [n for n in chart_tables if n not in tables]
    selected = st.selectbox("Table", names)
    source = tables.get(selected, chart_tables.get(selected))
    df = _table_to_dataframe(source, turn_dir)
    if df is None:
        st.json(source)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Cells", int(df.isna().sum().sum()))
    st.dataframe(df, width="stretch", height=520)


def _source_preview(source: str, *, max_lines: int = 40, max_chars: int = 4000) -> tuple[str, bool]:
    """Return a print-friendly source preview and whether anything was omitted."""
    if not source:
        return "", False
    lines = source.splitlines()
    by_lines = "\n".join(lines[:max_lines])
    preview = by_lines[:max_chars]
    omitted = len(lines) > max_lines or len(by_lines) > max_chars or len(source) > len(preview)
    return preview, omitted


def _artifact_widget_key(prefix: str, turn_dir: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(turn_dir)).strip("_")
    return f"{prefix}_{slug[-80:] or 'run'}"


def _render_artifacts(report: dict[str, Any], turn_dir: Path) -> None:
    with st.expander("Run artifacts and developer details", expanded=False):
        st.caption(
            "Technical artifacts are available for audit and reproduction. "
            "They are collapsed by default so the report stays focused on the scientific content."
        )

        code = report.get("code") or _read_text(turn_dir / "analysis.py") or ""
        preview, omitted = _source_preview(str(code))
        st.markdown("#### Generated analysis.py preview")
        if preview:
            st.caption("Preview only — first 40 lines or 4,000 characters, whichever comes first.")
            st.code(preview, language="python")
            if omitted:
                st.caption("Full source is available below or as a download.")
            st.download_button(
                "Download analysis.py",
                data=str(code),
                file_name="analysis.py",
                mime="text/x-python",
                key=_artifact_widget_key("download_analysis", turn_dir),
            )
            if st.checkbox("View full source", value=False, key=_artifact_widget_key("view_source", turn_dir)):
                st.code(str(code), language="python")
        else:
            st.caption("No generated analysis.py source was saved for this report.")

        with st.expander("stdout", expanded=False):
            st.text(_read_text(turn_dir / "stdout.txt") or report.get("stdout", ""))
        with st.expander("stderr", expanded=False):
            st.text(_read_text(turn_dir / "stderr.txt") or report.get("stderr", ""))
        with st.expander("Raw JSON report", expanded=False):
            st.json(report)


def _turn_label(turn: dict[str, Any]) -> str:
    report = turn["report"]
    status = "ok" if report.get("ok") else "failed"
    question = str(report.get("question") or "analysis")
    return f"{turn['name']} · {status} · {_truncate(question, 46)}"


def _table_to_dataframe(value: Any, turn_dir: Path) -> pd.DataFrame | None:
    if isinstance(value, str):
        path = _resolve_artifact_path(value, turn_dir)
        if path.exists() and path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return None
    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return None
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except Exception:
            try:
                return pd.DataFrame([value])
            except Exception:
                return None
    return None


def _resolve_artifact_path(value: Any, turn_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    # Relative path, or an absolute path that no longer exists (run dir moved to
    # another machine): fall back through the local layout by basename.
    candidates = [
        turn_dir / path if not path.is_absolute() else turn_dir / path.name,
        turn_dir / path.name,
        turn_dir / "figures" / path.name,
        turn_dir / "tables" / path.name,
        turn_dir / "sandbox" / path.name,
        turn_dir / "sandbox" / "tables" / path.name,
        turn_dir / "sandbox" / "figures" / path.name,
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _format_int(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Greyed placeholder for pipeline stages a run never reached
           (held-out verdicts / fix) — present but visibly dormant. */
        .ev-unavailable {
          border: 1.5px dashed var(--ev-border-strong);
          border-radius: var(--ev-radius);
          background: color-mix(in srgb, var(--ev-muted) 7%, var(--ev-panel));
          color: var(--ev-muted);
          padding: 1.1rem 1.3rem;
          margin: .4rem 0 1rem 0;
        }
        .ev-unavailable-title { font-weight: 600; margin-bottom: .35rem; }
        .ev-unavailable p { margin: .15rem 0; color: var(--ev-muted); }
        .ev-unavailable-hint { font-size: .86rem; opacity: .85; }
        /* Color values are the validated dataviz-skill reference palette
           (references/palette.md) plugged straight into these token names —
           status/accent hex are unchanged across light/dark by design, only
           surfaces/ink/soft-washes flip. Derived tints use color-mix() against
           --ev-panel/--ev-accent/etc. so they re-tint automatically in dark
           mode instead of needing a second hardcoded value. */
        :root {
          --ev-bg: #f9f9f7;
          --ev-panel: #fcfcfb;
          --ev-panel-elevated: #ffffff;
          --ev-border: #e1e0d9;
          --ev-border-strong: #98a2b3;
          --ev-text: #0b0b0b;
          --ev-text-secondary: #52514e;
          --ev-muted: #898781;
          --ev-accent: #2a78d6;
          --ev-accent-dark: #184f95;
          --ev-accent-soft: #cde2fb;
          --ev-ok: #0ca30c;
          --ev-warn: #fab219;
          --ev-serious: #ec835a;
          --ev-fail: #d03b3b;
          --ev-info: #2a78d6;
          --ev-radius: 12px;
          --ev-radius-sm: 8px;
          --ev-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 1px 3px rgba(16, 24, 40, 0.05);
          --ev-shadow-md: 0 6px 16px rgba(16, 24, 40, 0.08);
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --ev-bg: #0d0d0d;
            --ev-panel: #1a1a19;
            --ev-panel-elevated: #232322;
            --ev-border: #2c2c2a;
            --ev-border-strong: #4a4a47;
            --ev-text: #ffffff;
            --ev-text-secondary: #c3c2b7;
            --ev-muted: #898781;
            --ev-accent: #3987e5;
            --ev-accent-dark: #86b6ef;
            --ev-accent-soft: #0d366b;
            --ev-ok: #0ca30c;
            --ev-warn: #fab219;
            --ev-serious: #ec835a;
            --ev-fail: #d03b3b;
            --ev-info: #3987e5;
            --ev-shadow: 0 1px 2px rgba(0, 0, 0, 0.28), 0 1px 3px rgba(0, 0, 0, 0.32);
            --ev-shadow-md: 0 6px 16px rgba(0, 0, 0, 0.45);
          }
        }
        /* ---- Type scale -------------------------------------------------
           Tracking is size-specific: large text reads too loose as it grows
           and needs negative tracking, body sits near zero, small caps-y
           labels need a positive bump to stay legible. Leading moves
           inversely to size. Weight/size/leading are set together as one
           step, never size alone. */
        :root {
          --ev-track-display: -0.022em;
          --ev-track-title: -0.015em;
          --ev-track-body: 0em;
          --ev-track-label: 0.04em;
          --ev-lead-display: 1.08;
          --ev-lead-title: 1.2;
          --ev-lead-body: 1.55;
          /* Materials. Bigger surfaces read as thicker: more blur, deeper
             shadow. Shadows are layered (contact + ambient) rather than one
             soft blur, which is what keeps an edge crisp against the page. */
          --ev-material-thin: saturate(180%) blur(12px);
          --ev-material-thick: saturate(180%) blur(24px);
          --ev-surface-veil: rgba(255, 255, 255, 0.72);
          --ev-hairline: rgba(255, 255, 255, 0.5);
          --ev-shadow-lifted:
            0 0 0 0.5px rgba(16, 24, 40, 0.04),
            0 1px 2px rgba(16, 24, 40, 0.06),
            0 8px 24px -8px rgba(16, 24, 40, 0.12);
          --ev-ease-out: cubic-bezier(0.32, 0.72, 0, 1);
          --ev-dur-fast: 110ms;
          --ev-dur: 220ms;
        }
        @media (prefers-color-scheme: dark) {
          :root {
            --ev-surface-veil: rgba(26, 26, 25, 0.72);
            --ev-hairline: rgba(255, 255, 255, 0.08);
            --ev-shadow-lifted:
              0 0 0 0.5px rgba(0, 0, 0, 0.5),
              0 1px 2px rgba(0, 0, 0, 0.4),
              0 8px 24px -8px rgba(0, 0, 0, 0.6);
          }
        }
        /* System font first: it already ships optical sizing, tracking tables
           and legibility tuning that a webfont would have to re-earn. */
        html, body, [data-testid="stAppViewContainer"] {
          font-synthesis-weight: none;
          -webkit-font-smoothing: antialiased;
          text-rendering: optimizeLegibility;
        }
        .ev-header h1 {
          letter-spacing: var(--ev-track-display);
          line-height: var(--ev-lead-display);
          font-weight: 680;
        }
        .ev-report-answer-text, .ev-brief-value {
          letter-spacing: var(--ev-track-title);
          line-height: var(--ev-lead-title);
        }
        .ev-kicker, .ev-metric-label, .ev-brief-label {
          letter-spacing: var(--ev-track-label);
        }
        /* The verdict sentence sits directly under the headline: the reader
           should learn the outcome before reaching any evidence. Colour
           carries tone, but the sentence stands alone without it. */
        .ev-verdict-line {
          margin-top: 0.65rem;
          font-size: 0.98rem;
          font-weight: 600;
          letter-spacing: var(--ev-track-body);
          line-height: 1.45;
          padding-left: 0.7rem;
          border-left: 3px solid currentColor;
        }
        .ev-verdict-ok { color: var(--ev-ok); }
        .ev-verdict-partial { color: var(--ev-accent-dark); }
        .ev-verdict-null { color: var(--ev-text-secondary); }
        .ev-verdict-neutral { color: var(--ev-muted); }
        .ev-kicker .ev-pill { margin-left: 0.5rem; vertical-align: middle; }
        /* Numbers are the payload of this UI: lock them to tabular figures so
           a column of metrics aligns digit-for-digit instead of shimmering. */
        .ev-metric-value {
          font-variant-numeric: tabular-nums;
          letter-spacing: var(--ev-track-title);
        }
        /* ---- Materials --------------------------------------------------
           Chrome floats above content as a translucent layer rather than
           sitting in the flow behind a hard divider. */
        [data-testid="stSidebar"] {
          background: var(--ev-surface-veil) !important;
          backdrop-filter: var(--ev-material-thick);
          -webkit-backdrop-filter: var(--ev-material-thick);
          border-right: 1px solid var(--ev-hairline);
        }
        .ev-metric-card {
          transition:
            transform var(--ev-dur) var(--ev-ease-out),
            box-shadow var(--ev-dur) var(--ev-ease-out);
        }
        .ev-metric-card:hover {
          transform: translateY(-1px);
          box-shadow: var(--ev-shadow-lifted);
        }
        .ev-metric-card.ev-metric-lead {
          box-shadow: var(--ev-shadow-lifted);
        }
        /* ---- Response ---------------------------------------------------
           Feedback lands on pointer-DOWN, not on release: waiting for the
           click to complete is the single most common way an interface reads
           as laggy. */
        .stButton > button, [data-baseweb="tab"] {
          transition: transform var(--ev-dur-fast) var(--ev-ease-out),
                      background-color var(--ev-dur-fast) var(--ev-ease-out);
        }
        .stButton > button:active { transform: scale(0.975); }
        [data-baseweb="tab"]:active { transform: scale(0.99); }
        /* ---- Accessibility ----------------------------------------------
           Reduced motion means gentler feedback, not absent feedback; the
           transforms go, the colour response stays. */
        @media (prefers-reduced-motion: reduce) {
          .ev-metric-card, .stButton > button, [data-baseweb="tab"] {
            transition: none !important;
            transform: none !important;
          }
        }
        @media (prefers-reduced-transparency: reduce) {
          [data-testid="stSidebar"] {
            background: var(--ev-panel) !important;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
          }
        }
        @media (prefers-contrast: more) {
          .ev-metric-card, .ev-brief-card {
            border-color: var(--ev-text) !important;
            background: var(--ev-panel-elevated) !important;
          }
          .ev-metric-caption, .ev-metric-label { color: var(--ev-text-secondary) !important; }
        }
        /* Product chrome, not a debug tool: hamburger menu/Deploy button/Stop
           indicator, footer "Made with Streamlit" badge, and the top toolbar
           all hidden. --client.toolbarMode minimal (launch_dashboard) hides
           most of this server-side too; this CSS also covers direct
           `streamlit run` invocations that skip that flag. */
        #MainMenu, footer,
        [data-testid="stStatusWidget"], [data-testid="stDecoration"],
        [data-testid="stToolbar"], [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"], [data-testid="stMainMenuButton"] {
          visibility: hidden;
          height: 0;
        }
        /* The sidebar open/close chevrons live in the same header chrome the
           rules above purge. They must survive it: Streamlit auto-collapses
           the sidebar on narrow windows, and with the reopen button hidden a
           collapsed sidebar is unreachable forever ("the sidebar vanished").
           visibility:visible on the child overrides the hidden ancestor. */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
          visibility: visible !important;
          height: auto !important;
        }
        html, body, .stApp {
          font-family: -apple-system, "Segoe UI", "Inter", system-ui, sans-serif;
        }
        .stApp {
          background: var(--ev-bg);
          color: var(--ev-text);
        }
        [data-testid="stSidebar"] {
          background: var(--ev-panel-elevated);
          border-right: 1px solid var(--ev-border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: var(--ev-muted);
        }
        .block-container {
          /* Streamlit's fixed stHeader is ~3.75rem tall; anything smaller here
             lets the header clip the top of .ev-header (status pill/title get
             cut in half instead of just sitting close to it). */
          padding-top: 4rem;
          padding-bottom: 2.5rem;
          max-width: 1440px;
        }
        .ev-sidebar-title {
          color: var(--ev-text);
          font-size: 1.15rem;
          font-weight: 750;
          margin: 0.4rem 0 0.2rem;
        }
        .ev-header {
          align-items: flex-start;
          background: linear-gradient(135deg, var(--ev-panel-elevated) 0%, var(--ev-accent-soft) 145%);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-accent);
          border-radius: var(--ev-radius);
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          padding: 1rem 1.25rem;
          margin-bottom: 0.85rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-header h1 {
          color: var(--ev-text);
          font-size: 1.32rem;
          line-height: 1.3;
          margin: 0.15rem 0 0.3rem;
          font-weight: 780;
          letter-spacing: -0.01em;
        }
        .ev-kicker {
          color: var(--ev-accent-dark);
          font-size: 0.76rem;
          font-weight: 780;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        .ev-path {
          color: var(--ev-muted);
          font-size: 0.82rem;
          word-break: break-all;
        }
        .ev-overview-stage {
          color: var(--ev-text-secondary);
          font-size: 0.88rem;
          margin: -0.05rem 0 0.35rem;
        }
        .ev-header-right {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 0.45rem;
          min-width: 12rem;
        }
        .ev-pill {
          background: color-mix(in srgb, var(--ev-muted) 12%, var(--ev-panel));
          border: 1px solid var(--ev-border);
          border-radius: 999px;
          color: var(--ev-text);
          display: inline-block;
          font-size: 0.76rem;
          font-weight: 680;
          padding: 0.25rem 0.62rem;
          white-space: nowrap;
        }
        .ev-pill-ok {
          background: color-mix(in srgb, var(--ev-ok) 14%, var(--ev-panel));
          border-color: color-mix(in srgb, var(--ev-ok) 45%, var(--ev-panel));
          color: var(--ev-ok);
        }
        .ev-pill-fail {
          background: color-mix(in srgb, var(--ev-fail) 14%, var(--ev-panel));
          border-color: color-mix(in srgb, var(--ev-fail) 45%, var(--ev-panel));
          color: var(--ev-fail);
        }
        .ev-pill-warn {
          background: color-mix(in srgb, var(--ev-warn) 18%, var(--ev-panel));
          border-color: color-mix(in srgb, var(--ev-warn) 50%, var(--ev-panel));
          color: var(--ev-text);
        }
        .ev-pill-serious {
          background: color-mix(in srgb, var(--ev-serious) 16%, var(--ev-panel));
          border-color: color-mix(in srgb, var(--ev-serious) 45%, var(--ev-panel));
          color: var(--ev-serious);
        }
        .ev-metric-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          min-height: 6.8rem;
          padding: 0.8rem 0.85rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-metric-label {
          color: var(--ev-muted);
          font-size: 0.76rem;
          font-weight: 680;
          text-transform: uppercase;
        }
        .ev-metric-value {
          color: var(--ev-text);
          font-size: 1.8rem;
          font-weight: 780;
          line-height: 1.15;
          margin: 0.3rem 0;
        }
        .ev-metric-caption {
          color: var(--ev-muted);
          font-size: 0.78rem;
        }
        /* ---- uniform figure box ------------------------------------------
           Agent-authored figures arrive at whatever aspect ratio the analysis
           chose — a 7-panel violin grid next to a single bar chart. Rendering
           each at its native ratio makes a gallery of 23 look like debris.
           Every figure gets an identical frame and is fitted inside it.

           The frame lives on the WRAPPER, not the <img>: border and padding on
           the image itself inset the picture from the column edge while the
           title stayed flush, leaving every caption 7px adrift of the figure
           it labels. */
        [data-testid="stImage"] {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius-sm);
          /* No horizontal padding: the picture's left edge must sit on the
             same vertical as the caption above it. */
          padding: 0.45rem 0;
          margin-bottom: 0.3rem;
        }
        [data-testid="stImage"] img {
          width: 100%;
          aspect-ratio: 16 / 10;
          object-fit: contain;
          /* A figure narrower than 16:10 is pillarboxed by `contain`, which
             centres it and drifts it right of the caption. Pin it left. */
          object-position: left center;
          background: none;
          border: 0;
          padding: 0;
          border-radius: 0;
        }
        /* Titles of different lengths would otherwise start the images of a
           row at different heights. */
        .ev-card-title, .ev-card-subtitle {
          min-height: 2.6em;
          display: flex;
          align-items: flex-end;
          margin-bottom: 0.35rem;
        }
        /* ---- two panels, visibly two --------------------------------------
           Side-by-side figures with no boundary read as one continuous field,
           so the eye cannot tell where the left cell ends and the right
           begins. Any column that holds a figure becomes a card. Scoped with
           :has() so metric rows, forms and the sidebar are untouched. */
        [data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {
          gap: 1.15rem;
          margin-bottom: 1.15rem;
        }
        [data-testid="stColumn"]:has([data-testid="stImage"]) {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          padding: 0.95rem 1.05rem 0.6rem;
          box-shadow: var(--ev-shadow);
        }
        /* Inside a card the figure frame would be a second border around the
           same thing. It also has to disappear entirely: a figure wider than
           16:10 is letterboxed, and a contrasting frame turns that spare space
           into grey bands above and below the picture. */
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stImage"] {
          border-color: transparent;
          background: transparent;
          padding: 0;
        }
        /* The adjudicated verdict is the point of the whole run: give it the
           weight of a headline rather than letting it read as one count
           among many. */
        .ev-metric-card.ev-metric-lead {
          background: linear-gradient(180deg, rgba(10,123,188,0.09), rgba(10,123,188,0.02));
          border-color: rgba(10,123,188,0.42);
        }
        .ev-metric-lead .ev-metric-label { color: #0a7bbc; }
        .ev-metric-lead .ev-metric-value { font-size: 2.15rem; }
        .ev-brief-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin-bottom: 0.75rem;
        }
        .ev-brief-grid-two {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .ev-brief-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          min-height: 5rem;
          padding: 0.75rem 0.85rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-brief-label {
          color: var(--ev-accent);
          font-size: 0.75rem;
          font-weight: 760;
          margin-bottom: 0.45rem;
          text-transform: uppercase;
        }
        .ev-brief-value {
          color: var(--ev-text);
          font-size: 0.92rem;
          line-height: 1.45;
        }
        .ev-report-answer {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-accent);
          border-radius: var(--ev-radius);
          margin-bottom: 0.9rem;
          padding: 1rem 1.1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-report-answer-text {
          color: var(--ev-text);
          font-size: 1rem;
          line-height: 1.45;
        }
        .ev-stage-map {
          display: grid;
          gap: 0.55rem;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          margin: 0.2rem 0 0.95rem;
        }
        .ev-stage-card {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          min-height: 3.4rem;
          padding: 0.55rem 0.65rem;
          margin-bottom: 0.65rem;
        }
        .ev-stage-active {
          border-color: color-mix(in srgb, var(--ev-accent) 45%, var(--ev-panel));
          box-shadow: inset 0 0 0 2px var(--ev-accent-soft);
        }
        .ev-stage-id {
          color: var(--ev-accent);
          font-size: 0.78rem;
          font-weight: 800;
          margin-bottom: 0.25rem;
        }
        .ev-stage-name {
          color: var(--ev-text);
          font-size: 0.82rem;
          font-weight: 760;
          line-height: 1.25;
        }
        .ev-stage-strip {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
          margin: 0.1rem 0 0.75rem;
        }
        .ev-stage-chip {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-radius: 999px;
          color: var(--ev-muted);
          display: inline-flex;
          font-size: 0.78rem;
          gap: 0.25rem;
          line-height: 1.2;
          padding: 0.32rem 0.65rem;
          white-space: nowrap;
        }
        .ev-stage-chip.ev-stage-active {
          background: var(--ev-accent-soft);
          border-color: color-mix(in srgb, var(--ev-accent) 45%, var(--ev-panel));
          color: var(--ev-accent);
        }
        .ev-analysis-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          min-height: 9.2rem;
          padding: 0.95rem 1rem;
          margin-bottom: 0.8rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-analysis-method,
        .ev-analysis-evidence,
        .ev-analysis-takeaway {
          color: var(--ev-text-secondary);
          font-size: 0.9rem;
          line-height: 1.45;
          margin-top: 0.35rem;
        }
        .ev-analysis-takeaway {
          border-left: 3px solid var(--ev-ok);
          background: color-mix(in srgb, var(--ev-ok) 8%, var(--ev-panel));
          padding: 0.45rem 0.6rem;
        }
        .ev-storyboard-card {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-ok);
          border-radius: var(--ev-radius);
          margin: 0.35rem 0 0.85rem;
          padding: 0.9rem 1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-storyboard-title {
          color: var(--ev-text);
          font-size: 1rem;
          font-weight: 760;
          margin-bottom: 0.35rem;
        }
        .ev-storyboard-summary {
          color: var(--ev-text-secondary);
          font-size: 0.92rem;
          line-height: 1.45;
        }
        .ev-structure-card {
          background: var(--ev-panel-elevated);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-accent);
          border-radius: var(--ev-radius);
          margin: 0.35rem 0 0.65rem;
          padding: 0.85rem 1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-structure-outcome {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 730;
          margin-top: 0.35rem;
        }
        .ev-evidence-summary {
          background: color-mix(in srgb, var(--ev-ok) 8%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-ok) 35%, var(--ev-panel));
          border-left: 4px solid var(--ev-ok);
          border-radius: var(--ev-radius);
          margin: 0.25rem 0 0.75rem;
          padding: 0.85rem 1rem;
        }
        .ev-evidence-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-info);
          border-radius: var(--ev-radius);
          margin: 0.55rem 0;
          padding: 0.85rem 1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-evidence-card.ev-evidence-audit {
          border-left-color: var(--ev-border-strong);
          background: var(--ev-panel);
        }
        .ev-evidence-card-top {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          justify-content: space-between;
          margin-bottom: 0.35rem;
        }
        .ev-evidence-name {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
        }
        .ev-evidence-pill {
          background: color-mix(in srgb, var(--ev-ok) 14%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-ok) 35%, var(--ev-panel));
          border-radius: 999px;
          color: var(--ev-ok);
          font-size: 0.72rem;
          font-weight: 760;
          padding: 0.15rem 0.5rem;
          text-transform: uppercase;
        }
        .ev-evidence-meaning {
          color: var(--ev-text-secondary);
          font-size: 0.92rem;
          line-height: 1.45;
          margin-bottom: 0.5rem;
        }
        .ev-evidence-takeaway {
          background: color-mix(in srgb, var(--ev-accent) 6%, var(--ev-panel));
          border: 1px solid var(--ev-border);
          border-left: 3px solid var(--ev-info);
          border-radius: var(--ev-radius);
          color: var(--ev-text-secondary);
          font-size: 0.92rem;
          line-height: 1.45;
          margin: 0.7rem 0 0.75rem;
          padding: 0.7rem 0.85rem;
        }
        .ev-evidence-meta {
          color: var(--ev-muted);
          display: flex;
          flex-wrap: wrap;
          font-size: 0.78rem;
          gap: 0.45rem 0.75rem;
        }
        .ev-claim-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          margin: 0.7rem 0 0.35rem;
          padding: 0.9rem 1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-claim-supported {
          border-left: 4px solid var(--ev-ok);
        }
        .ev-claim-inconclusive,
        .ev-claim-descriptive {
          border-left: 4px solid var(--ev-border-strong);
        }
        .ev-claim-refuted {
          border-left: 4px solid var(--ev-fail);
        }
        .ev-claim-top {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
          margin-bottom: 0.5rem;
        }
        .ev-claim-text {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
          line-height: 1.35;
          margin-bottom: 0.4rem;
        }
        .ev-note {
          align-items: flex-start;
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          display: flex;
          gap: 0.75rem;
          margin-bottom: 0.65rem;
          padding: 0.85rem;
        }
        .ev-note-index {
          align-items: center;
          background: var(--ev-accent-soft);
          border: 1px solid color-mix(in srgb, var(--ev-accent) 40%, var(--ev-panel));
          border-radius: 999px;
          color: var(--ev-accent);
          display: flex;
          flex: 0 0 1.7rem;
          font-size: 0.78rem;
          font-weight: 760;
          height: 1.7rem;
          justify-content: center;
        }
        .ev-signal {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          margin-bottom: 0.75rem;
          padding: 0.95rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-signal-title {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
          margin-bottom: 0.35rem;
        }
        .ev-signal-body {
          color: var(--ev-text-secondary);
          font-size: 0.88rem;
          margin-bottom: 0.55rem;
        }
        .ev-signal-test {
          background: color-mix(in srgb, var(--ev-accent) 6%, var(--ev-panel));
          border-left: 3px solid var(--ev-accent);
          color: var(--ev-text-secondary);
          font-size: 0.82rem;
          padding: 0.45rem 0.6rem;
        }
        .ev-card-title {
          color: var(--ev-text);
          font-size: 0.95rem;
          font-weight: 740;
          margin: 0.3rem 0 0.45rem;
        }
        .ev-card-subtitle {
          color: var(--ev-text);
          font-size: 0.9rem;
          font-weight: 720;
          margin: 0.6rem 0 0.25rem;
        }
        .ev-section-head {
          border-bottom: 1px solid var(--ev-border);
          margin: 0.1rem 0 1.1rem;
          padding-bottom: 0.6rem;
        }
        .ev-section-title {
          color: var(--ev-text);
          font-size: 1.32rem;
          font-weight: 780;
          letter-spacing: -0.01em;
          line-height: 1.3;
        }
        .ev-section-sub {
          color: var(--ev-muted);
          font-size: 0.86rem;
          margin-top: 0.2rem;
        }
        .ev-findings-overview {
          background: color-mix(in srgb, var(--ev-accent) 5%, var(--ev-panel));
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          box-shadow: var(--ev-shadow);
          margin: 0.85rem 0 1rem;
          padding: 1rem 1.1rem;
        }
        .ev-findings-overview-kicker {
          color: var(--ev-accent-dark);
          font-size: 0.78rem;
          font-weight: 780;
          letter-spacing: 0.04em;
          margin-bottom: 0.15rem;
          text-transform: uppercase;
        }
        .ev-findings-overview-sub {
          color: var(--ev-muted);
          font-size: 0.85rem;
          margin-bottom: 0.75rem;
        }
        .ev-finding-index-row {
          align-items: flex-start;
          border-top: 1px solid color-mix(in srgb, var(--ev-border) 75%, transparent);
          display: flex;
          gap: 0.7rem;
          padding: 0.65rem 0 0.1rem;
        }
        .ev-finding-index-num {
          align-items: center;
          background: color-mix(in srgb, var(--ev-accent) 12%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-accent) 36%, var(--ev-panel));
          border-radius: 999px;
          color: var(--ev-accent-dark);
          display: flex;
          flex: 0 0 1.55rem;
          font-size: 0.78rem;
          font-weight: 780;
          height: 1.55rem;
          justify-content: center;
          width: 1.55rem;
        }
        .ev-finding-index-title {
          color: var(--ev-text);
          font-size: 0.95rem;
          font-weight: 720;
          line-height: 1.35;
        }
        .ev-finding-index-meta {
          color: var(--ev-muted);
          font-size: 0.8rem;
          margin-top: 0.12rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-takeaway-head) {
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          box-shadow: var(--ev-shadow);
          margin-bottom: 1.1rem;
          padding: 1.15rem 1.3rem 1.3rem;
          transition: box-shadow 120ms ease, border-color 120ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ev-takeaway-head):hover {
          border-color: color-mix(in srgb, var(--ev-accent) 35%, var(--ev-panel));
          box-shadow: var(--ev-shadow-md);
        }
        .ev-takeaway-head {
          align-items: center;
          display: flex;
          gap: 0.7rem;
          margin-bottom: 0.15rem;
        }
        .ev-takeaway-badge {
          /* Always darkens from --ev-accent (not --ev-accent-dark, which is
             deliberately a *lighter* blue in dark mode for text-legibility
             elsewhere) so the white glyph stays readable in both modes. */
          align-items: center;
          background: linear-gradient(135deg, var(--ev-accent), color-mix(in srgb, var(--ev-accent) 100%, black 35%));
          border-radius: 999px;
          box-shadow: 0 2px 6px rgba(42, 120, 214, 0.35);
          color: #ffffff;
          display: flex;
          flex: 0 0 2rem;
          font-size: 0.95rem;
          font-weight: 800;
          height: 2rem;
          justify-content: center;
          width: 2rem;
        }
        .ev-takeaway-title {
          color: var(--ev-text);
          font-size: 1.08rem;
          font-weight: 760;
          line-height: 1.35;
        }
        .ev-takeaway-evidence-empty {
          color: var(--ev-muted);
          font-size: 0.82rem;
          font-style: italic;
          margin: 0.5rem 0;
        }
        .ev-evidence-reading {
          background: color-mix(in srgb, var(--ev-accent) 5%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-accent) 18%, var(--ev-border));
          border-radius: var(--ev-radius-sm);
          color: var(--ev-text-secondary);
          font-size: 0.9rem;
          line-height: 1.45;
          margin: 0.55rem 0 0.35rem;
          padding: 0.62rem 0.72rem;
        }
        .ev-evidence-reading-label {
          color: var(--ev-accent-dark);
          font-size: 0.72rem;
          font-weight: 780;
          letter-spacing: 0.04em;
          margin-bottom: 0.18rem;
          text-transform: uppercase;
        }
        .ev-evidence-caution {
          background: color-mix(in srgb, var(--ev-warn) 10%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-warn) 34%, var(--ev-panel));
          border-radius: var(--ev-radius-sm);
          color: var(--ev-text-secondary);
          font-size: 0.82rem;
          line-height: 1.4;
          margin: 0.35rem 0 0.6rem;
          padding: 0.45rem 0.65rem;
        }
        .ev-evidence-caution span {
          color: var(--ev-text);
          font-weight: 760;
          margin-right: 0.25rem;
        }
        .ev-takeaway-analysis {
          background: color-mix(in srgb, var(--ev-accent) 6%, var(--ev-panel));
          border-left: 3px solid var(--ev-accent);
          border-radius: 0 var(--ev-radius-sm) var(--ev-radius-sm) 0;
          color: var(--ev-text-secondary);
          font-size: 0.94rem;
          line-height: 1.6;
          margin-top: 1rem;
          padding: 0.75rem 0.95rem;
        }
        .ev-takeaway-caveat {
          /* --ev-warn is sub-3:1 on light surfaces (validated palette note) —
             text stays primary ink; the warning color only tints the wash. */
          background: color-mix(in srgb, var(--ev-warn) 14%, var(--ev-panel));
          border: 1px solid color-mix(in srgb, var(--ev-warn) 45%, var(--ev-panel));
          border-radius: var(--ev-radius-sm);
          color: var(--ev-text);
          font-size: 0.82rem;
          margin-top: 0.6rem;
          padding: 0.45rem 0.75rem;
        }
        div[data-testid="stMetric"] {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius);
          padding: 0.65rem 0.75rem;
          box-shadow: var(--ev-shadow);
        }
        div[data-testid="stMetricValue"] {
          font-size: 1.45rem;
          line-height: 1.15;
        }
        div[data-testid="stMetricLabel"] {
          font-size: 0.78rem;
        }
        div[data-testid="stTabs"] {
          border-bottom: 1px solid var(--ev-border);
          margin-bottom: 0.9rem;
        }
        div[data-testid="stTabs"] button {
          font-size: 0.92rem;
          font-weight: 700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
          color: var(--ev-accent-dark);
        }
        /* stImage is deliberately absent here — its frame is defined once, in
           the uniform-figure-box block above. Restating it later at higher
           specificity is what knocked every figure 7px out of line with its
           own title. */
        div[data-testid="stDataFrame"],
        div[data-testid="stVegaLiteChart"] {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: var(--ev-radius-sm);
          padding: 0.4rem;
        }
        h3 {
          color: var(--ev-text);
          font-size: 1.05rem;
          margin-top: 1rem;
        }
        .ev-hero-task {
          color: var(--ev-text-secondary);
          font-size: 0.92rem;
          line-height: 1.4;
          margin-top: 0.3rem;
        }
        .ev-hero-verdict {
          align-items: center;
          display: flex;
          gap: 0.6rem;
          margin: 0.35rem 0 0.15rem;
        }
        .ev-hero-verdict-text {
          color: var(--ev-text);
          font-size: 1.02rem;
          font-weight: 700;
          line-height: 1.35;
        }
        .ev-kpi-row {
          display: grid;
          gap: 0.65rem;
          grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
          margin: 0.7rem 0 0.85rem;
        }
        .ev-step {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-accent);
          border-radius: var(--ev-radius);
          margin-bottom: 0.7rem;
          padding: 0.85rem 1rem;
          box-shadow: var(--ev-shadow);
        }
        .ev-step-rejected {
          border-left-color: var(--ev-warn);
        }
        .ev-step-top {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          justify-content: space-between;
          margin-bottom: 0.35rem;
        }
        .ev-step-index {
          color: var(--ev-muted);
          font-size: 0.78rem;
          font-weight: 760;
          text-transform: uppercase;
        }
        .ev-step-tool {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
        }
        .ev-step-rationale {
          color: var(--ev-text-secondary);
          font-size: 0.92rem;
          line-height: 1.45;
          margin-bottom: 0.35rem;
        }
        .ev-step-outcome {
          color: var(--ev-text-secondary);
          font-size: 0.85rem;
        }
        @media (max-width: 900px) {
          .ev-header {
            flex-direction: column;
          }
          .ev-header-right {
            justify-content: flex-start;
          }
          .ev-brief-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
