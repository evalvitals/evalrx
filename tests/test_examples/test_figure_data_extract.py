"""Contract checks for examples/benchmark/tools/extract_figure_data.py.

This tool turns a run directory into the numbers a paper figure prints, so the
failure mode that matters is a *plausible wrong number*: a bar chart whose
failure rates are computed over the wrong denominator, a held-out validation
whose 2x2 does not add up, a summary key silently renamed in ``_common/runner.py``
so the header strip goes blank. None of that raises — it just ships. Each test
below pins one of those numbers on a synthetic run whose answers are known by
construction.

The last three cover the `case-study-figure` skill that draws from this data —
in particular that every `qa_flags` the tool can emit has a stated consequence on
the figure, since a caveat with no rule is a caveat the figure quietly omits.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "examples" / "benchmark" / "tools"
_RUNNER = Path(__file__).resolve().parents[2] / "examples" / "benchmark" / "_common" / "runner.py"

# The keys extract_figure_data reads out of summary.json (see block_run). The
# producer is _common/runner.py; test_summary_contract_with_the_runner guards
# the join.
CONSUMED_SUMMARY_KEYS = frozenset({
    "model", "spec", "dataset", "modality", "backend",
    "n_cases", "baseline_accuracy", "cycles", "stopped_by",
})


@pytest.fixture(scope="module")
def efd():
    sys.path.insert(0, str(_TOOL))
    try:
        spec = importlib.util.spec_from_file_location(
            "extract_figure_data", _TOOL / "extract_figure_data.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_TOOL))


# --------------------------------------------------------------------------
# a synthetic run whose every number is known by construction
# --------------------------------------------------------------------------

# six cases: four explore (two fail), two held-out. The signal is discrete with
# three levels, and it is perfectly aligned with the label at the top level.
SIGNAL = "termination_audit.continuation_chars"
EXPLORE = [
    # (case id, label, signal value)
    ("c-1", "fail", 2),
    ("c-2", "fail", 2),
    ("c-3", "pass", 1),
    ("c-4", "pass", 0),
]
HELDOUT = ["c-5", "c-6"]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_run(root: Path, *, signal_values=None, m4_effect=-0.42, n_broken=1) -> Path:
    """Write a run directory in the RunLoggerV2 shape _common/runner.py +
    RunContext produce: run.json (run-wide events + manifest) plus one JSON
    document per stage under M1..M5, no run_log.jsonl / manifest.json /
    report/*.json siblings."""
    rows = signal_values if signal_values is not None else EXPLORE
    logs = root / "logs"

    _write(root / "summary.json", {
        "model": "gemma-4-e2b", "spec": "gemma-4-e2b-it", "dataset": "chartqa",
        "modality": "vlm", "backend": "hf_local", "n_cases": len(rows) + len(HELDOUT),
        "baseline_accuracy": 0.5, "cycles": 1, "stopped_by": "converged",
        "n_verified": 1, "fix": {"recommendation": "L2", "attempted": []},
    })

    discovery = [
        {"id": cid, "prompt": f"question {cid}", "expected": "42",
         "observed": "43" if label == "fail" else "42", "label": label}
        for cid, label, _ in rows
    ] + [
        {"id": cid, "prompt": f"question {cid}", "expected": "42",
         "observed": "42", "label": "pass"} for cid in HELDOUT
    ]

    # four numeric fields, one per measurement-inventory verdict: the signal
    # itself varies (candidate), strict_match is a function of the answer key,
    # n_options never varies, conf_logprob covers only half the cases
    per_case = [
        dict({"sample_id": cid, "continuation_chars": value,
              "strict_match": 0 if label == "fail" else 1, "n_options": 4},
             **({"conf_logprob": -0.1 * i} if i < len(rows) // 2 else {}))
        for i, (cid, label, value) in enumerate(rows)
    ]

    stats = [
        {"tool": "signal_label_assoc", "config": {"signal": SIGNAL}, "effect": m4_effect,
         "p_value": 0.001, "fdr_corrected": True, "correction_method": "BH",
         "correction_family": "m2_assoc", "summary": f"assoc CI={m4_effect:+.4f}..+0.0100",
         "details": {"n_signal": 2, "n_control": 2,
                     "fail_rate_signal": 1.0, "fail_rate_control": 0.0}},
        {"tool": "signal_label_assoc", "config": {"signal": "other.flag"}, "effect": 0.01,
         "p_value": 0.9, "fdr_corrected": False, "correction_method": "BH",
         "correction_family": "m2_assoc", "summary": "degenerate: one level only"},
    ]

    cases = [
        {"case_id": cid,
         "case": {"label": label, "inputs": {"prompt": f"question {cid}"}},
         "media_paths": ["artifacts/case_media/x.png"] if label == "fail" else []}
        for cid, label, _ in rows
    ]

    _write(logs / "run.json", {
        "run_start": {"judge": "claude", "data_fingerprint": "abc123",
                      "label_distribution": {"fail": 2, "pass": 2}},
        "manifest": {
            "run_id": "logs",
            "config": {"fix_tier": "L2", "m1_selection": "auto", "confirm_split": 0.33},
        },
        "cases": cases,
        "diagnose_reports": [{"discovery": discovery}],
        "loop_end": [{"total_duration_sec": 12.5}],
    })
    _write(logs / "M1" / "log.json", {"probe": [
        {"cycle": 0,
         "selected_analyzers": ["termination_audit", "logit_lens"],
         "selection_rationale": "text run: termination first",
         "findings": {"termination_audit": {"continuation_rate": 0.5}},
         "results": {"termination_audit": {"findings": {"per_case": per_case}}}},
    ]})
    _write(logs / "M2" / "log.json", {"analysis": [
        {"cycle": 0, "stats_results": stats},
        {"cycle": -1, "stats_results": stats[:1]},
    ]})
    _write(logs / "M3" / "log.json", {"diagnosis": [
        {"hypotheses": [
            {"failure_mode": "runaway_generation", "statement": "the model never stops",
             "test_design": "continuation_chars vs label", "expected_direction": "higher"}],
         "n_critic_kept": 1, "n_critic_rejected": 0, "review": {"objection": "confounded"}},
    ]})
    _write(logs / "M4" / "log.json", {"surgery": [
        {"module": "m4", "failure_mode": "runaway_generation",
         "status": "supported", "hypothesis": "the model never stops",
         "evidence": {"m4_test_name": "signal_label_assoc", "expected_direction": "higher",
                      "effect_size": m4_effect, "ci": [-0.6, -0.2], "reject": True,
                      "m4_evidence_grade": "B"}},
    ]})
    _write(logs / "M5" / "log.json", {"fix": [
        {"best": {
            "name": "stop_sequences", "tier": "L2", "n_pairs": 10,
            "n_baseline_correct": 4, "n_candidate_correct": 6,
            "baseline_rate": 0.4, "candidate_rate": 0.6,
            "n_fixed": 3, "n_broken": n_broken, "effect": 0.2,
            "summary": "paired CI=+0.0500..+0.3500", "verdict": "accept"}},
    ]})
    return root


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path / "chartqa.chain1")


def blocks(records, name):
    return [r for r in records if r["block"] == name]


def one(records, name):
    found = blocks(records, name)
    assert len(found) == 1, f"expected exactly one {name} block, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_summary_contract_with_the_runner():
    """Every summary.json key the tool reads must be one the runner still writes."""
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    written: set[str] = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "summary" and isinstance(node.value, ast.Dict):
            written |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert written, "no summary dict literal found in _common/runner.py"
    missing = CONSUMED_SUMMARY_KEYS - written
    assert not missing, f"extract_figure_data reads keys the runner no longer writes: {missing}"


def test_run_block_carries_the_header_strip(efd, run_dir):
    run = one(efd.extract(str(run_dir)), "run")
    assert run["model"] == "gemma-4-e2b"
    assert run["model_spec"] == "gemma-4-e2b-it"
    assert run["dataset"] == "chartqa"
    assert run["modality"] == "vlm"
    assert run["backend"] == "hf_local"
    assert run["baseline_accuracy"] == 0.5
    assert run["stopped_by"] == "converged"
    assert run["judge"] == "claude"
    # the split is inferred: explore = cases with a case_record, held-out = the rest
    assert (run["n_explore"], run["n_heldout"]) == (4, 2)
    assert run["n_cases"] == 6
    assert run["run_id"] == "chartqa.chain1"


def test_signal_curve_is_binned_by_level_with_the_right_denominators(efd, run_dir):
    curve = one(efd.extract(str(run_dir)), "m1_signal_curve")
    assert curve["signal"] == SIGNAL
    assert curve["binning"] == "levels"
    assert curve["phase"] == "explore"
    assert curve["n_cases"] == 4, "only the explore split has labels to bin"
    by_value = {b["value"]: b for b in curve["bins"]}
    assert [b["value"] for b in curve["bins"]] == [0, 1, 2], "levels must be sorted"
    assert (by_value[2]["n_cases"], by_value[2]["n_fail"]) == (2, 2)
    assert by_value[2]["failure_rate"] == 1.0
    assert by_value[1]["failure_rate"] == 0.0
    assert by_value[0]["failure_rate"] == 0.0


def test_continuous_signal_is_quartile_binned_and_flagged(efd, tmp_path):
    rows = [(f"c-{i}", "fail" if i % 3 == 0 else "pass", float(i)) for i in range(20)]
    root = make_run(tmp_path / "cont", signal_values=rows)
    records = efd.extract(str(root))
    curve = one(records, "m1_signal_curve")
    assert curve["binning"] == "quartiles"
    assert curve["n_distinct_values"] == 20
    assert sum(b["n_cases"] for b in curve["bins"]) == 20, "every case lands in exactly one bin"
    assert all(b["failure_rate"] is not None for b in curve["bins"])
    codes = {f["code"] for f in one(records, "qa_flags")["flags"]}
    assert "signal_binned_for_plotting" in codes


def test_the_measurement_funnel_says_why_each_field_was_dropped(efd, run_dir):
    """A dropped measurement must be dropped for a stated reason, not silently."""
    probe = one(efd.extract(str(run_dir)), "m1_probe_questions")
    status = {m["field"]: m["status"] for m in probe["inventory"]}
    assert status["continuation_chars"] == "candidate"
    assert status["strict_match"] == "dropped_sees_answer_key", (
        "a field computed from the answer key would let a signal predict the label from the label")
    assert status["n_options"] == "dropped_never_varies"
    assert status["conf_logprob"] == "dropped_partial_coverage"
    assert probe["n_measured"] == 4
    assert probe["dropped"] == {"saw_the_answer_key": 1, "never_varied": 1, "partial_coverage": 1}
    # the funnel's other end is M2: signals that entered the correction family
    assert probe["n_forwarded"] == 1


def test_probe_questions_are_plain_language_and_exclude_the_answer_key(efd, run_dir):
    probe = one(efd.extract(str(run_dir)), "m1_probe_questions")
    asked = [q["question"] for q in probe["questions"]]
    assert "Did it stop, or keep talking?" in asked, "continuation_chars asks this"
    assert "Was it as sure as it sounded?" in asked, "conf_logprob asks this"
    # strict_match is answer-key-derived, so it is not a question the probe asks
    assert all("answer key" not in q for q in asked)
    assert len(asked) == len(set(asked)), "one question per row, analyzers merged"
    # the reading order is fixed, so the list is stable across runs
    assert asked == sorted(asked, key=lambda q: efd.QUESTION_ORDER.index(q))
    stop = next(q for q in probe["questions"] if q["question"] == "Did it stop, or keep talking?")
    assert stop["analyzers"] == ["termination_audit"]
    assert stop["n_candidates"] == 1


def test_m2_family_separates_explore_from_heldout(efd, run_dir):
    records = efd.extract(str(run_dir))
    fams = {f["phase"]: f for f in blocks(records, "m2_family")}
    assert set(fams) == {"explore", "heldout"}
    assert fams["explore"]["survivors"] == [SIGNAL]
    # the degenerate result is reported but kept out of the multiplicity family
    assert fams["explore"]["n_signal_label_assoc"] == 2
    assert fams["explore"]["n_degenerate"] == 1
    assert fams["explore"]["n_in_correction_family"] == 1
    degenerate = [t for t in blocks(records, "m2_test") if t["degenerate"]]
    assert degenerate and all(not t["in_correction_family"] for t in degenerate)


def test_validation_2x2_adds_up_and_recovers_the_ci(efd, run_dir):
    val = one(efd.extract(str(run_dir)), "validation")
    assert val["phase"] == "heldout"
    assert val["candidate"] == "stop_sequences"
    # n_pairs = both_correct + both_wrong + fixed + broken, by construction
    assert val["both_correct"] == 3          # 4 baseline-correct minus the 1 broken
    assert val["both_wrong"] == 3            # 10 - 3 - 3 - 1
    assert (val["both_correct"] + val["both_wrong"]
            + val["n_fixed"] + val["n_broken"]) == val["n_pairs"]
    assert val["ci"] == pytest.approx([0.05, 0.35]), "CI is parsed out of the summary string"


def test_qa_flags_catch_a_direction_mismatch_and_a_regressing_fix(efd, run_dir):
    # the fixture's hypothesis expects "higher" but the observed effect is negative
    codes = {f["code"] for f in one(efd.extract(str(run_dir)), "qa_flags")["flags"]}
    assert "direction_mismatch" in codes
    assert "accepted_fix_breaks_cases" in codes
    assert "example_case_from_explore" in codes
    assert "correction_family_size_differs" not in codes


def test_a_clean_run_raises_no_flags(efd, tmp_path):
    root = make_run(tmp_path / "clean", m4_effect=0.42, n_broken=0)
    records = efd.extract(str(root))
    codes = {f["code"] for f in blocks(records, "qa_flags") for f in f["flags"]}
    assert "direction_mismatch" not in codes
    assert "accepted_fix_breaks_cases" not in codes


def test_example_case_prefers_a_failing_case_with_media(efd, run_dir):
    ex = one(efd.extract(str(run_dir)), "example_case")
    assert ex["label"] == "fail"
    assert ex["split"] == "explore"
    assert ex["media_paths"] == ["logs/artifacts/case_media/x.png"]
    assert ex["case_id"] in {"c-1", "c-2"}


def test_forced_example_case_id_is_honoured_and_unknown_ids_are_skipped(efd, run_dir):
    assert one(efd.extract(str(run_dir), "c-3"), "example_case")["case_id"] == "c-3"
    assert not blocks(efd.extract(str(run_dir), "nope"), "example_case")


def test_a_broken_block_does_not_lose_the_rest(efd, run_dir):
    """Per-block failures are caught, so one bad stage document cannot empty
    the figure."""
    (run_dir / "logs" / "M2" / "log.json").write_text("{ not json", encoding="utf-8")
    records = efd.extract(str(run_dir))
    assert one(records, "run")["model"] == "gemma-4-e2b"
    assert blocks(records, "validation")


def test_find_run_roots_accepts_a_run_or_a_parent_of_runs(efd, tmp_path):
    parent = tmp_path / "runs"
    first = make_run(parent / "chartqa.chain1")
    second = make_run(parent / "mmau.chain1")
    assert efd.find_run_roots(str(first)) == [str(first)]
    assert efd.find_run_roots(str(parent)) == [str(first), str(second)]


def test_document_and_markdown_render_the_run(efd, run_dir):
    doc = efd.to_document(efd.extract(str(run_dir)))
    assert doc["run"]["model"] == "gemma-4-e2b"
    assert doc["headline"]["baseline_accuracy"] == 0.5
    assert doc["m1_probe"]["signal_curve"]["signal"] == SIGNAL
    assert doc["m1_probe"]["measurement_inventory"], "the funnel travels with the document"
    assert doc["m4_verdicts"]["verdicts"][0]["status"] == "supported"
    assert doc["_sources"], "every section must say which file it came from"
    # the figure draws five cards in this order, named by the pipeline block
    assert [p["module"] for p in doc["pipeline"]] == ["M1", "M2", "M3", "M4", "M5"]
    assert doc["pipeline"][0]["name"] == "Suspicious Behavior Detection"
    md = efd.to_markdown(doc, title="chartqa.chain1")
    assert "chartqa.chain1" in md
    assert SIGNAL in md
    assert "**40.0% → 60.0%**" in md, "the headline transition is the figure's caption"
    assert "SUPPORTED" in md
    assert "What the probes ask" in md
    assert "4 measurements, 1 forwarded to M2" in md
    for name in efd.MODULE_NAMES.values():
        assert name in md


def test_trial_root_is_reanchored_on_the_run_root(efd, tmp_path):
    """The run log records the WRITER's absolute path — in a container that is
    /app/work/outputs/<run>/logs/..., which exists on no host. relpath against
    it walked up to / and emitted a ../../.. chain whose length depended on
    where the reader sat. The artifact lives inside the run dir: re-anchor."""
    root = make_run(tmp_path / "chartqa.chain1")
    trial = root / "logs" / "M5" / "artifacts" / "05_L2_stop_sequences"
    trial.mkdir(parents=True)
    m5_path = root / "logs" / "M5" / "log.json"
    doc = json.loads(m5_path.read_text())

    def _set_trial_root(path: str) -> None:
        doc["fix"][-1]["best"]["trial_root"] = path
        doc["fix"][-1]["best"]["payload"] = {"name": "stop_sequences", "strategy": "single"}
        m5_path.write_text(json.dumps(doc))

    _set_trial_root("/app/work/outputs/chartqa.chain1/logs/M5/artifacts/05_L2_stop_sequences")
    recs = efd.extract(str(root))
    repair = next(r for r in recs if r["block"] == "repair")
    assert repair["trial_root"] == "logs/M5/artifacts/05_L2_stop_sequences"

    # a recorded path whose logs/ suffix does NOT exist under this root keeps
    # the old relpath fallback (it may genuinely live elsewhere)
    _set_trial_root("/somewhere/else/logs/fixes/99_missing")
    recs = efd.extract(str(root))
    repair = next(r for r in recs if r["block"] == "repair")
    assert "99_missing" in repair["trial_root"] and not repair["trial_root"].startswith("logs/")
# --------------------------------------------------------------------------
# the case-study-figure skill, which draws from what the tool extracts
# --------------------------------------------------------------------------

_SKILL = _TOOL / "case-study-figure"


def _frontmatter(text: str) -> dict:
    assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    out, key = {}, None
    for line in block.splitlines():
        if line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
        elif key:
            out[key] += " " + line.strip()
    return out


def test_the_skill_is_a_well_formed_agent_skill():
    meta = _frontmatter((_SKILL / "SKILL.md").read_text(encoding="utf-8"))
    assert meta["name"] == _SKILL.name, "the skill's name must match its directory"
    assert meta["version"] and meta["description"]
    # bundled_skill_paths() only picks up a directory with a SKILL.md at its root
    assert (_SKILL / "SKILL.md").is_file()


def test_every_qa_flag_the_tool_emits_is_binding_on_the_figure():
    """A new flag with no rule in the skill is a caveat the figure would omit."""
    emitted = set(re.findall(r'"code": "([a-z_]+)"',
                             (_TOOL / "extract_figure_data.py").read_text(encoding="utf-8")))
    assert len(emitted) >= 8, "expected the tool's qa_flags to be found by name"
    skill = (_SKILL / "SKILL.md").read_text(encoding="utf-8")
    missing = {code for code in emitted if code not in skill}
    assert not missing, f"qa_flags with no consequence stated in SKILL.md: {missing}"


def test_the_skill_only_points_at_files_that_exist():
    for doc in (_SKILL / "SKILL.md", _SKILL / "references" / "figure-spec.md"):
        for target in re.findall(r"\]\((?!https?:)([^)#]+)\)", doc.read_text(encoding="utf-8")):
            assert (doc.parent / target).resolve().exists(), f"{doc.name} -> {target}"
    # step 1 of the skill runs the extractor by this path
    assert "extract_figure_data.py" in (_SKILL / "SKILL.md").read_text(encoding="utf-8")
    for asset in ("casestudy_chartqa.svg", "casestudy_mmau.svg", "qualitative_vlm_L2.pdf"):
        assert (_SKILL / "references" / asset).is_file()
