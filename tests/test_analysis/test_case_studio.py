"""Case-study view for paper-method bench runs: manifest join, flip labels,
repair-method description, and the dashboard tab that renders them.

The loader half is Streamlit-free and always runs; the render half is guarded
on the dashboard extras like the other dashboard tests.
"""

from __future__ import annotations

import json
import struct
import sys

import pytest

from evalvitals.analysis import case_studio as cs


def _wav_bytes(seconds: float = 0.01, rate: int = 8000) -> bytes:
    """Smallest valid mono 16-bit PCM WAV — enough for st.audio to accept."""
    frames = int(rate * seconds)
    data = b"\x00\x00" * frames
    return (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(data)) + data
    )


def _build_audio_bench(root, *, with_pointer: bool = True, with_l2: bool = False):
    """An MMAU-shaped example dir: data/<manifest>.jsonl + audio/ + outputs/."""
    data = root / "data"
    (data / "audio").mkdir(parents=True)
    outputs = root / "outputs"
    outputs.mkdir()

    rows = []
    for index in range(6):
        case_id = f"case-{index}"
        (data / "audio" / f"{case_id}.wav").write_bytes(_wav_bytes())
        rows.append({
            "id": case_id,
            "instruction": f"What made the sound in clip {index}?",
            "choices": ["(A) a siren", "(B) a piano", "(C) a dog", "(D) a bell"],
            "expected": "A",
            "audio_path": f"audio/{case_id}.wav",
            "duration_sec": 10.0,
            "task": "multiple_choice",
            "metadata": {"mmau_task": "sound" if index % 2 else "music",
                         "difficulty": "easy"},
        })
    (data / "mmau_test_mini.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )

    # selection split: 4 cases, 2 correct. tcd repairs case-0, breaks nothing;
    # attend_carefully repairs nothing and breaks case-1.
    selection_cases = [
        {"id": "case-0", "output": "D", "correct": False},
        {"id": "case-1", "output": "A", "correct": True},
        {"id": "case-2", "output": "A", "correct": True},
        {"id": "case-3", "output": "C", "correct": False},
    ]
    report = {
        "paper": "tcd",
        "backend": "hf_local",
        "model": "qwen2-audio-7b-instruct",
        "paper_method_fidelity": "native_layer_matched_stability",
        "max_tier": "L3a",
        "splits": {"diagnosis": 0, "selection": 4, "confirmation": 2, "shuffle_seed": 1},
        "baseline": {
            "diagnosis": {"n": 0, "correct": 0, "accuracy": 0.0},
            "selection": {"n": 4, "correct": 2, "accuracy": 0.5, "cases": selection_cases},
            "confirmation": {"n": 2, "correct": 1, "accuracy": 0.5, "cases": [
                {"id": "case-4", "output": "B", "correct": False},
                {"id": "case-5", "output": "A", "correct": True},
            ]},
        },
        "hypothesis": {"statement": "Transient acoustic cues are under-weighted.",
                       "predicted_failure_mode": "temporal_smoothing_bias"},
        "auto_fix": {
            "selection": {
                "max_tier": "L3a",
                "attempted": [
                    {"tier": "L1", "name": "attend_carefully", "kind": "template",
                     "source": "default",
                     "payload": {"prompt_template": "Examine the image carefully. {prompt}"},
                     "n_pairs": 4, "n_fixed": 0, "n_broken": 1, "effect": -0.25,
                     "e_value": 3.0, "verdict": "regressed", "summary": "[mcnemar] regressed",
                     "fixed_cases": [], "broken_cases": ["case-1"]},
                    # Full runner-shaped summary line (the CI lives only here),
                    # unlike attend_carefully's terse one above — the UI has to
                    # handle both.
                    {"tier": "L3a", "name": "tcd_temporal_blur", "kind": "tcd",
                     "source": "paper_default", "payload": {},
                     "n_pairs": 4, "n_fixed": 1, "n_broken": 0, "effect": 0.25,
                     "e_value": 2.0, "verdict": "partial",
                     "summary": "[mcnemar + e-value (paired binary)] effect=+0.2500 (B>A) "
                                "CI=+0.0000..+0.5000, e=2.00 -> inconclusive "
                                "[partial, coverage=100%]",
                     "coverage": 1.0, "n_unstable": 0,
                     "fixed_cases": ["case-0"], "broken_cases": []},
                ],
                "best": None,
                "recommendation": {"recommend_tier": "L4", "reason": "nothing validated"},
                "ebh_survivors": [],
            },
            "confirmation": {"skipped": "no selection candidate"},
        },
    }
    if with_l2:
        # A real L2 spec: prompt untouched (`{prompt}`), the intervention is
        # entirely in the multi-call strategy.
        report["auto_fix"]["selection"]["attempted"].insert(1, {
            "tier": "L2", "name": "self_refine", "kind": "spec", "source": "default",
            "payload": {"name": "self_refine", "image_ops": [],
                        "prompt_template": "{prompt}", "n_samples": 1,
                        "generation_kwargs": {}, "strategy": "self_refine",
                        "output_key_pattern": ""},
            "n_pairs": 4, "n_fixed": 0, "n_broken": 2, "effect": -0.5,
            "e_value": 8.0, "verdict": "regressed", "summary": "[mcnemar] regressed",
            "fixed_cases": [], "broken_cases": ["case-2", "case-3"],
        })
    if with_pointer:
        report["dataset"] = {"manifest": "data/mmau_test_mini.jsonl",
                             "media_field": "audio_path", "media_kind": "audio"}
    (outputs / "tcd_mmau.json").write_text(json.dumps(report), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_bench_report_detection_requires_per_case_rows():
    assert not cs.is_bench_report({"baseline": {}, "auto_fix": {}})
    assert not cs.is_bench_report({"observations": []})
    assert cs.is_bench_report({
        "baseline": {"selection": {"cases": [{"id": "a"}]}}, "auto_fix": {},
    })


@pytest.mark.parametrize("with_pointer", [True, False])
def test_case_study_joins_manifest_media_and_flips(tmp_path, with_pointer):
    """With or without the report's `dataset` pointer, the questions, options,
    .wav files and per-candidate flips all resolve."""
    _build_audio_bench(tmp_path, with_pointer=with_pointer)
    (study,) = [
        cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)
    ]

    assert study.manifest_coverage == 1.0
    assert study.media_kind == "audio"
    assert len(study.cases) == 6

    repaired = study.case_by_id("case-0")
    assert repaired.question.startswith("What made the sound")
    assert repaired.choices[0] == ("A", "a siren")
    assert repaired.expected == "A" and repaired.expected_text == "a siren"
    assert repaired.media.kind == "audio" and repaired.media.exists
    assert repaired.baseline_correct is False and repaired.baseline_output == "D"
    assert repaired.flips == {"attend_carefully": cs.FLIP_UNCHANGED,
                              "tcd_temporal_blur": cs.FLIP_REPAIRED}
    assert repaired.flipped_by() == ["tcd_temporal_blur"]

    broken = study.case_by_id("case-1")
    assert broken.broken_by() == ["attend_carefully"]

    # The sweep only ran on the selection split, so a confirmation case must not
    # be labelled "unchanged by" a candidate that never touched it.
    assert study.case_by_id("case-4").flips == {}


def test_confirmation_cases_are_never_labelled_by_the_sweep(tmp_path):
    _build_audio_bench(tmp_path)
    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]
    assert all(not c.flips for c in study.cases if c.split == "confirmation")
    assert all(c.flips for c in study.cases if c.split == "selection")


def test_missing_manifest_degrades_to_outcomes_with_a_note(tmp_path):
    _build_audio_bench(tmp_path)
    (tmp_path / "data" / "mmau_test_mini.jsonl").unlink()
    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]

    assert study.manifest_path is None
    assert all(c.unresolved and not c.question for c in study.cases)
    assert any("No benchmark manifest was found" in n for n in study.notes)


def test_manifest_is_chosen_by_case_id_coverage(tmp_path):
    """A data dir holding several benchmarks' manifests (vlm_paper_benchmark)
    must resolve to the one that actually covers this report's ids."""
    _build_audio_bench(tmp_path, with_pointer=False)
    (tmp_path / "data" / "aaa_other_benchmark.jsonl").write_text(
        json.dumps({"id": "unrelated-0", "question": "?", "expected": "A"}) + "\n",
        encoding="utf-8",
    )
    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]
    assert study.manifest_path.name == "mmau_test_mini.jsonl"


def test_image_rows_resolve_as_image_media(tmp_path):
    """The VLM slices use question/options/image — the same loader must read
    them without the runner renaming anything."""
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "images" / "0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (data / "pope.jsonl").write_text(json.dumps({
        "id": "pope-0", "question": "Is there a cat?", "options": ["Yes", "No"],
        "expected": "A", "image": "images/0.png", "task": "yes_no", "metadata": {},
    }) + "\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "pope_hf.json").write_text(json.dumps({
        "paper": "vcd", "model": "llava",
        "baseline": {"selection": {"n": 1, "correct": 0, "accuracy": 0.0,
                                   "cases": [{"id": "pope-0", "output": "No", "correct": False}]}},
        "auto_fix": {"selection": {"attempted": [], "best": None}},
    }), encoding="utf-8")

    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]
    case = study.case_by_id("pope-0")
    assert study.media_kind == "image"
    assert case.media.kind == "image" and case.media.exists
    assert case.question == "Is there a cat?"
    # Unmarked options still get positional letters so "A"/"B" mean something.
    assert case.choices == [("A", "Yes"), ("B", "No")]


def test_split_choice_only_strips_real_option_markers():
    assert cs.split_choice("(A) siren", 0) == ("A", "siren")
    assert cs.split_choice("B. piano", 1) == ("B", "piano")
    assert cs.split_choice("C) dog", 2) == ("C", "dog")
    # A bare answer that merely starts with a letter must stay intact.
    assert cs.split_choice("A minor chord", 3) == ("D", "A minor chord")


def test_describe_candidate_explains_the_mechanism_and_knobs():
    template = cs.describe_candidate({
        "tier": "L1", "name": "attend_carefully", "kind": "template",
        "payload": {"prompt_template": "Look closely. {prompt}"},
    })
    assert "Prompt template" in template["note"]
    assert template["prompt_template"] == "Look closely. {prompt}"
    assert template["knobs"] == {}

    scaffold = cs.describe_candidate({
        "tier": "L2", "name": "self_refine", "kind": "spec",
        "payload": {"name": "self_refine", "strategy": "self_refine",
                    "prompt_template": "{prompt}", "n_samples": 1, "image_ops": []},
    })
    assert "critique" in scaffold["note"]
    assert scaffold["knobs"] == {"n_samples": 1}

    paper = cs.describe_candidate({
        "tier": "L3a", "name": "tcd_temporal_blur", "kind": "tcd", "payload": {},
    })
    assert "Temporal Contrastive Decoding" in paper["note"]
    assert paper["defaults_only"] is True


def test_accuracy_by_facet_groups_the_baseline_arm(tmp_path):
    _build_audio_bench(tmp_path)
    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]
    rows = cs.accuracy_by_facet(study.cases, "mmau_task")
    assert {r["value"] for r in rows} == {"sound", "music"}
    assert all(0.0 <= r["accuracy"] <= 1.0 and r["n"] > 0 for r in rows)


def test_case_label_hides_the_outcome_in_blind_mode(tmp_path):
    _build_audio_bench(tmp_path)
    (study,) = [cs.build_case_study(p) for p in cs.find_bench_reports(tmp_path)]
    case = study.case_by_id("case-0")
    assert cs.case_label(case).startswith("🛠")
    assert cs.case_label(case, blind=True).startswith("•")


def test_load_run_reports_a_casebench_run(tmp_path):
    from evalvitals.analysis.dashboard import load_run

    _build_audio_bench(tmp_path)
    session = load_run(tmp_path)
    assert session["kind"] == "casebench"
    assert len(session["case_studies"]) == 1
    assert session["runs"] == [] and session["story"] is None


# ---------------------------------------------------------------------------
# Dashboard rendering (needs the dashboard extras)
# ---------------------------------------------------------------------------


def _run_app(run_dir):
    pytest.importorskip("streamlit")
    pytest.importorskip("pandas")
    from streamlit.testing.v1 import AppTest

    from evalvitals.analysis import dashboard_app

    sys.argv = ["dashboard_app.py", str(run_dir)]
    # Absolute: AppTest resolves a relative script path against the *calling*
    # file, which is this test module, not the repo root.
    at = AppTest.from_file(dashboard_app.__file__, default_timeout=60)
    at.run()
    return at


def test_dashboard_renders_the_case_study_tabs(tmp_path):
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Run Overview", "2 Case Study", "3 Repair Methods",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Case study — play it, answer it" in blob
    assert "Repair methods — what was actually changed" in blob
    assert "Transient acoustic cues are under-weighted." in blob
    # The audio element is what makes a human able to answer the item at all.
    assert len(at.get("audio")) == 1
    # Blind mode is on by default, so the question is shown but no answer is.
    assert any("What made the sound" in str(m.value) for m in at.markdown)
    assert "Correct answer:" not in blob


def test_case_study_reveals_answers_and_repair_flips_when_unblinded(tmp_path):
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    at.toggle(key="ev_cs_blind::tcd_mmau").set_value(False).run()
    assert not at.exception

    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Correct answer:" in blob
    assert "What the model answered" in blob
    # case-0 is selected first and tcd repaired it.
    flips = [df.value for df in at.dataframe]
    assert any("repaired this case" in str(v.values) for v in flips)


def test_locking_in_an_answer_scores_the_human_against_the_model(tmp_path):
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    at.radio(key="ev_cs_pick::tcd_mmau::case-0").set_value("A").run()
    at.button(key="ev_cs_lock::tcd_mmau::case-0").click().run()
    assert not at.exception

    labels = {m.label: m.value for m in at.metric}
    assert labels["You"] == "1/1"          # the human picked the right option
    assert labels["Model, same cases"] == "0/1"  # the baseline got it wrong
    assert labels["Gap"] == "+1"


def test_outcome_filter_is_disabled_in_blind_mode(tmp_path):
    """Filtering by "cases the model got wrong" would hand the reader the
    answer before they attempt the item."""
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    blinded = [s for s in at.selectbox if s.label == "Outcome filter"][0]
    assert blinded.options == ["Every case"]

    at.toggle(key="ev_cs_blind::tcd_mmau").set_value(False).run()
    unblinded = [s for s in at.selectbox if s.label == "Outcome filter"][0]
    assert "Broken by some candidate" in unblinded.options

    unblinded.set_value("broke").run()
    picker = [s for s in at.selectbox if s.label.startswith("Case (")][0]
    assert picker.label == "Case (1 shown)"
    assert at.session_state["ev_cs_case::tcd_mmau"] == "case-1"

    # Re-blinding must not leave the widget holding an option that no longer exists.
    at.toggle(key="ev_cs_blind::tcd_mmau").set_value(True).run()
    assert not at.exception
    assert at.session_state["ev_cs_outcome::tcd_mmau"] == "all"


def test_repair_methods_tab_jumps_to_a_flipped_case(tmp_path):
    """The Repair Methods tab lists the case ids a candidate flipped; clicking
    one has to select it in the Case Study tab, clear the filters that would
    otherwise hide it, AND move the reader to that tab — a preselected case the
    reader is not looking at is not a jump."""
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)
    at.toggle(key="ev_cs_blind::tcd_mmau").set_value(False).run()
    [s for s in at.selectbox if s.label == "Outcome filter"][0].set_value("repaired").run()

    at.button(key="ev_cs_jump::tcd_mmau::attend_carefully::case-1").click().run()

    assert not at.exception
    assert at.session_state["ev_cs_case::tcd_mmau"] == "case-1"
    assert at.session_state["ev_cs_outcome::tcd_mmau"] == "all"
    assert at.session_state["ev_cs_tab"] == "2 Case Study"


def test_repair_methods_tab_shows_the_prompt_and_the_mechanism(tmp_path):
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Temporal Contrastive Decoding" in blob
    assert "Prompt template (L1)" in blob
    # The template reads as a diff, not a flat code block: the wording this
    # repair injected is greened, the benchmark's own prompt stays a slot. That
    # is what makes "the image" on an audio benchmark jump out.
    assert '<span class="ev-prompt-added">Examine the image carefully. </span>' in blob
    assert '<span class="ev-prompt-slot">' in blob
    assert not any("Examine the image carefully" in str(c.value) for c in at.code)


def test_verdict_legend_covers_every_branch_fix_agent_can_emit():
    """The legend explains FixAgent's verdicts, so it must not drift from them.

    Walks the real ``_verdict`` over the whole (reject x effect x fixed x
    broken) grid and asserts the UI has a card, a table colour and a markdown
    colour for everything it can return."""
    pytest.importorskip("streamlit")
    pytest.importorskip("pandas")
    from evalvitals.eval_agent.stages.fix_agent import FixAgent, FixValidation

    from evalvitals.analysis.dashboard_app import (
        _FIX_VERDICT_GUIDE, _FIX_VERDICT_HEX, _FIX_VERDICT_MD,
    )

    emitted = set()
    for reject in (True, False):
        for effect in (-0.5, 0.0, 0.5):
            for n_fixed in range(3):
                for n_broken in range(3):
                    v = FixValidation(candidate=None, n_fixed=n_fixed, n_broken=n_broken,
                                      effect=effect, reject=reject)
                    v.fixed = v.reject and (v.effect or 0.0) > 0
                    emitted.add(FixAgent._verdict(v))

    explained = {name for name, _, _, _ in _FIX_VERDICT_GUIDE}
    assert emitted <= explained, f"legend misses verdict(s): {sorted(emitted - explained)}"
    # `not_executed` is assigned outside _verdict (no scorable pair / crash).
    assert "not_executed" in explained
    assert explained <= set(_FIX_VERDICT_HEX)
    assert explained <= set(_FIX_VERDICT_MD)


def test_verdict_legend_counts_this_runs_candidates(tmp_path):
    """The legend doubles as a summary, so the per-verdict count has to be the
    run's own tally — not a static key."""
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "ev-verdict-legend" in blob
    # This fixture has exactly one regressed (attend_carefully) and one partial
    # (tcd_temporal_blur); the other verdicts must render as an empty card.
    for name, count in (("regressed", 1), ("partial", 1)):
        assert (f'<span class="ev-verdict-name">{name}</span>'
                f'<span class="ev-verdict-count">{count}</span>') in blob
    for name in ("fixed", "no_effect", "unsafe", "not_executed"):
        assert (f'<span class="ev-verdict-name">{name}</span>'
                f'<span class="ev-verdict-count">0</span>') in blob
        assert "ev-verdict-card-empty" in blob
    # The rule text is the point of the legend, not the colour.
    assert "rejects H0 AND effect &gt; 0" in blob


def test_verdict_column_is_colour_coded(tmp_path):
    """The verdict column carries a per-verdict tint so the table is scannable."""
    pytest.importorskip("pandas")
    from evalvitals.analysis.dashboard_app import _FIX_VERDICT_HEX, _verdict_cell_style

    assert "#0ca30c" in _verdict_cell_style("fixed")
    assert "rgba(12,163,12,0.16)" in _verdict_cell_style("fixed")
    assert _FIX_VERDICT_HEX["regressed"] in _verdict_cell_style("regressed")
    assert _verdict_cell_style("fixed") != _verdict_cell_style("unsafe")
    # An unknown label from some other runner must not be styled into a lie.
    assert _verdict_cell_style("something_else") == ""

    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    assert at.dataframe, "the candidate table should still render"


def test_repair_methods_card_shows_no_raw_statistics(tmp_path):
    """The card describes the intervention; the verdict legend and the table's
    repaired/broke counts carry the outcome. Neither the runner's raw summary
    line nor a re-formatted CI/e-value belongs on it."""
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    for leaked in ("REJECT H0", "CI ", "e-value", "effect ", "paired binary",
                   "of the paired split", "[mcnemar]"):
        assert leaked not in captions, f"statistics leaked back onto the card: {leaked!r}"


def test_strategy_flow_matches_what_the_pipeline_sends():
    """The diagram is read off fix_tools, so it must stay true to the pipeline.

    Each step's tooltip is the instruction that call really prepends, and the
    steps are distinct prompts — not one prompt repeated per call."""
    from evalvitals.eval_agent.stages import fix_tools

    # case_studio mirrors the table instead of importing it (analysis may not
    # depend on eval_agent at runtime), so this equality IS the sync mechanism.
    assert cs.STRATEGY_CALLS == dict(fix_tools.STRATEGY_CALLS)

    refine = cs.strategy_steps("self_refine")
    assert [label for label, _ in refine] == ["answer", "critique that answer", "revise it"]
    prompts = [prompt for _, prompt in refine]
    assert prompts[0] == ""                       # sends the task prompt as-is
    assert len({p for p in prompts if p}) == 2    # critique != revise
    assert "concise correction advice only" in prompts[1]
    assert "Do not discuss the revision process" in prompts[2]

    # least_to_most makes TWO calls, not three: it never answers the
    # subproblems individually, it solves the original given the decomposition.
    assert [label for label, _ in cs.strategy_steps("least_to_most")] == [
        "break into subproblems", "solve using the decomposition",
    ]

    # Every strategy the pipeline admits (besides `direct`) is drawable.
    named = set(fix_tools._SCAFFOLD_STRATEGIES) - {"direct"}
    assert named == set(fix_tools.STRATEGY_CALLS)

    # The hover reveal is wired by enumerated `data-i` CSS rules, which only go
    # up to 6 — a longer chain would render steps whose prompt never shows.
    assert max(len(calls) for calls in fix_tools.STRATEGY_CALLS.values()) <= 6

    # `self_consistency_5` is not a strategy at all — it is `direct` with
    # n_samples=5, so its chain comes from the sample count.
    assert cs.strategy_steps("", n_samples=1) == []
    assert [label for label, _ in cs.strategy_steps("", n_samples=5)] == [
        "sample 5 answers independently", "take the majority answer",
    ]


def test_l2_card_shows_the_strategy_not_the_tier_blurb(tmp_path):
    """An L2 card must lead with what THIS candidate does, not a paragraph
    restating what L2 is — and must not frame an untouched prompt as a diff."""
    _build_audio_bench(tmp_path, with_l2=True)
    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Input/scaffold spec (L2)" not in blob          # generic blurb dropped
    assert "answer once, critique that answer" in blob     # the specific thing kept
    # The multi-call shape IS the intervention, so it is drawn — each step
    # paired by data-i with a reveal row holding that call's own instruction.
    for index, label in enumerate(("answer", "critique that answer", "revise it"), 1):
        assert f'<span class="ev-flow-step" data-i="{index}">{label}</span>' in blob
        assert f'<div class="ev-flow-tip" data-i="{index}">' in blob
    assert "Sends the task prompt as-is, with nothing prepended." in blob
    assert "Give concise correction advice only." in blob
    assert "Do not discuss the revision process." in blob
    # No title= tooltip on the chips: it is browser-delayed and drops on a
    # fast pointer. (The phrase also appears in the injected CSS comment
    # explaining why, so scope the check to the chip markup itself.)
    import re as _re
    assert not _re.search(r'<span class="ev-flow-step"[^>]*title=', blob)
    # An untouched prompt gets no section at all — not an empty box, and not a
    # line of prose saying nothing happened.
    assert "Prompt unchanged" not in blob
    # L1's real wrapper still gets its diff box on the same page.
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Prompt unchanged" not in captions
    assert "Green is text this repair ADDED" in captions


def test_candidate_table_is_scannable(tmp_path):
    """The table shows a tier handle and the outcome, nothing else: no blank
    marker column, no statistics competing with the verdict for attention."""
    pytest.importorskip("pandas")
    from evalvitals.analysis.dashboard_app import _candidate_handles

    assert _candidate_handles([
        {"tier": "L1"}, {"tier": "L2"}, {"tier": "L2"}, {"tier": "L3a"},
    ]) == ["L1_1", "L2_1", "L2_2", "L3a_1"]

    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    # The verdict column is styled, so this element carries a Styler; other
    # tabs render plain frames. Unwrap and pick the candidate table by shape.
    frames = [getattr(d.value, "data", d.value) for d in at.dataframe]
    matching = [f for f in frames if "candidate" in list(f.columns)]
    assert matching, f"no candidate table among {[list(f.columns) for f in frames]}"
    frame = matching[0]
    assert list(frame.columns) == [
        "candidate", "kind", "source", "repaired", "broke", "pairs", "verdict",
    ]
    assert list(frame["candidate"]) == ["L1_1", "L3a_1"]
    # The card below is what maps a handle back to a name — and carries its
    # verdict in colour so a collapsed card still reads.
    labels = [e.label for e in at.expander]
    assert "L1_1 · attend_carefully — :red[**regressed**]" in labels
    assert "L3a_1 · tcd_temporal_blur — :orange[**partial**]" in labels
