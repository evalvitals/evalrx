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


def _build_audio_bench(root, *, with_pointer: bool = True):
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
                    {"tier": "L3a", "name": "tcd_temporal_blur", "kind": "tcd",
                     "source": "paper_default", "payload": {},
                     "n_pairs": 4, "n_fixed": 1, "n_broken": 0, "effect": 0.25,
                     "e_value": 2.0, "verdict": "partial", "summary": "[mcnemar] inconclusive",
                     "fixed_cases": ["case-0"], "broken_cases": []},
                ],
                "best": None,
                "recommendation": {"recommend_tier": "L4", "reason": "nothing validated"},
                "ebh_survivors": [],
            },
            "confirmation": {"skipped": "no selection candidate"},
        },
    }
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
    one has to select it in the Case Study tab and clear the filters that would
    otherwise hide it."""
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)
    at.toggle(key="ev_cs_blind::tcd_mmau").set_value(False).run()
    [s for s in at.selectbox if s.label == "Outcome filter"][0].set_value("repaired").run()

    at.button(key="ev_cs_jump::tcd_mmau::attend_carefully::case-1").click().run()

    assert not at.exception
    assert at.session_state["ev_cs_case::tcd_mmau"] == "case-1"
    assert at.session_state["ev_cs_outcome::tcd_mmau"] == "all"


def test_repair_methods_tab_shows_the_prompt_and_the_mechanism(tmp_path):
    _build_audio_bench(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Temporal Contrastive Decoding" in blob
    assert "Prompt template (L1)" in blob
    assert any("Examine the image carefully. {prompt}" in str(c.value) for c in at.code)
