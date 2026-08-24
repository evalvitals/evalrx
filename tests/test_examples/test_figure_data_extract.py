"""Contract checks for examples/benchmark/tools/extract_figure_data.py.

This tool turns a run directory into the numbers a paper figure prints, so the
failure mode that matters is a *plausible wrong number*: a bar chart whose
failure rates are computed over the wrong denominator, a held-out validation
whose 2x2 does not add up, a summary key silently renamed in ``_common/runner.py``
so the header strip goes blank. None of that raises — it just ships. Each test
below pins one of those numbers on a synthetic run whose answers are known by
construction.
"""

from __future__ import annotations

import ast
import importlib.util
import json
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


def make_run(root: Path, *, signal_values=None, m5_effect=-0.42, n_broken=1) -> Path:
    """Write a run directory in the shape _common/runner.py + RunContext produce."""
    rows = signal_values if signal_values is not None else EXPLORE
    logs = root / "logs"

    _write(root / "summary.json", {
        "model": "gemma-4-e2b", "spec": "gemma-4-e2b-it", "dataset": "chartqa",
        "modality": "vlm", "backend": "hf_local", "n_cases": len(rows) + len(HELDOUT),
        "baseline_accuracy": 0.5, "cycles": 1, "stopped_by": "converged",
        "n_verified": 1, "fix": {"recommendation": "L2", "attempted": []},
    })
    _write(logs / "manifest.json", {
        "run_id": "logs",
        "config": {"fix_tier": "L2", "m1_selection": "auto", "confirm_split": 0.33},
    })
    _write(logs / "report" / "discovery_cases.json", [
        {"id": cid, "prompt": f"question {cid}", "expected": "42",
         "observed": "43" if label == "fail" else "42", "label": label}
        for cid, label, _ in rows
    ] + [
        {"id": cid, "prompt": f"question {cid}", "expected": "42",
         "observed": "42", "label": "pass"} for cid in HELDOUT
    ])
    _write(logs / "artifacts" / "c0_termination_audit.result.json", {
        "findings": {"per_case": [
            {"sample_id": cid, "continuation_chars": value} for cid, _, value in rows
        ]},
    })

    stats = [
        {"tool": "signal_label_assoc", "config": {"signal": SIGNAL}, "effect": m5_effect,
         "p_value": 0.001, "fdr_corrected": True, "correction_method": "BH",
         "correction_family": "m2_assoc", "summary": f"assoc CI={m5_effect:+.4f}..+0.0100",
         "details": {"n_signal": 2, "n_control": 2,
                     "fail_rate_signal": 1.0, "fail_rate_control": 0.0}},
        {"tool": "signal_label_assoc", "config": {"signal": "other.flag"}, "effect": 0.01,
         "p_value": 0.9, "fdr_corrected": False, "correction_method": "BH",
         "correction_family": "m2_assoc", "summary": "degenerate: one level only"},
    ]
    _write(logs / "artifacts" / "c0_m2_stats_results.json", stats)
    _write(logs / "artifacts" / "post_m2_stats_results.json", stats[:1])

    events = [
        {"event": "run_start", "judge": "claude", "data_fingerprint": "abc123",
         "label_distribution": {"fail": 2, "pass": 2}},
        {"event": "probe", "cycle": 0,
         "selected_analyzers": ["termination_audit", "logit_lens"],
         "selection_rationale": "text run: termination first",
         "findings": {"termination_audit": {"continuation_rate": 0.5}}},
    ]
    events += [
        {"event": "case_record", "case_id": cid,
         "case": {"label": label, "inputs": {"prompt": f"question {cid}"}},
         "media_paths": ["artifacts/case_media/x.png"] if label == "fail" else []}
        for cid, label, _ in rows
    ]
    events += [
        {"event": "diagnosis", "hypotheses": [
            {"failure_mode": "runaway_generation", "statement": "the model never stops",
             "test_design": "continuation_chars vs label", "expected_direction": "higher"}],
         "n_critic_kept": 1, "n_critic_rejected": 0, "review": {"objection": "confounded"}},
        {"event": "surgery", "module": "m5", "failure_mode": "runaway_generation",
         "status": "supported", "hypothesis": "the model never stops",
         "evidence": {"m5_test_name": "signal_label_assoc", "expected_direction": "higher",
                      "effect_size": m5_effect, "ci": [-0.6, -0.2], "reject": True,
                      "m5_evidence_grade": "B"}},
        {"event": "fix", "best": {
            "name": "stop_sequences", "tier": "L2", "n_pairs": 10,
            "n_baseline_correct": 4, "n_candidate_correct": 6,
            "baseline_rate": 0.4, "candidate_rate": 0.6,
            "n_fixed": 3, "n_broken": n_broken, "effect": 0.2,
            "summary": "paired CI=+0.0500..+0.3500", "verdict": "accept"}},
        {"event": "loop_end", "total_duration_sec": 12.5},
    ]
    (logs / "run_log.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
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
    root = make_run(tmp_path / "clean", m5_effect=0.42, n_broken=0)
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
    """Per-block failures are caught, so one bad artifact cannot empty the figure."""
    (run_dir / "logs" / "artifacts" / "c0_m2_stats_results.json").write_text(
        "{ not json", encoding="utf-8")
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
    assert doc["m5_verdicts"][0]["status"] == "supported"
    assert doc["_sources"], "every section must say which file it came from"
    md = efd.to_markdown(doc, title="chartqa.chain1")
    assert "chartqa.chain1" in md
    assert SIGNAL in md
    assert "**40.0% → 60.0%**" in md, "the headline transition is the figure's caption"
    assert "SUPPORTED" in md


def test_trial_root_is_reanchored_on_the_run_root(efd, tmp_path):
    """The run log records the WRITER's absolute path — in a container that is
    /app/work/outputs/<run>/logs/..., which exists on no host. relpath against
    it walked up to / and emitted a ../../.. chain whose length depended on
    where the reader sat. The artifact lives inside the run dir: re-anchor."""
    root = make_run(tmp_path / "chartqa.chain1")
    trial = root / "logs" / "fixes" / "05_L2_stop_sequences"
    trial.mkdir(parents=True)
    lines = (root / "logs" / "run_log.jsonl").read_text().splitlines()
    events = [json.loads(x) for x in lines]
    for e in events:
        if e.get("event") == "fix":
            e["best"]["trial_root"] = "/app/work/outputs/chartqa.chain1/logs/fixes/05_L2_stop_sequences"
            e["best"]["payload"] = {"name": "stop_sequences", "strategy": "single"}
    (root / "logs" / "run_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    recs = efd.extract(str(root))
    repair = next(r for r in recs if r["block"] == "repair")
    assert repair["trial_root"] == "logs/fixes/05_L2_stop_sequences"

    # a recorded path whose logs/ suffix does NOT exist under this root keeps
    # the old relpath fallback (it may genuinely live elsewhere)
    for e in events:
        if e.get("event") == "fix":
            e["best"]["trial_root"] = "/somewhere/else/logs/fixes/99_missing"
    (root / "logs" / "run_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    recs = efd.extract(str(root))
    repair = next(r for r in recs if r["block"] == "repair")
    assert "99_missing" in repair["trial_root"] and not repair["trial_root"].startswith("logs/")
