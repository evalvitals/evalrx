"""Contract checks for the case-study sheet the served report renders.

The sheet is the one place a reader meets the whole run at once, so its failure
mode is a number that is plausible and wrong: a BH denominator that disagrees
with the forest plot it labels, a funnel that counts a signal computed from the
answer key, a ladder that reports "untouched" for a tier the search actually
climbed. Each test below pins one of those against a run whose answers are known
by construction.

The reference for the numbers is the CLI tool at
``examples/benchmark/tools/extract_figure_data.py``: the two are supposed to
read the same artifacts the same way, and the last test runs both over one
synthetic run and holds their numbers to each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalrx.reporting.case_study import build_case_study

SIGNAL = "coverage_verification_gap.n_unique"
# four explore cases, two of them failing, and the signal is aligned with the
# label at its top level
EXPLORE = [("c-1", "fail", 5), ("c-2", "fail", 5), ("c-3", "pass", 1), ("c-4", "pass", 2)]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_run(root: Path, *, n_broken: int = 1, verdict_status: str = "supported") -> Path:
    """A run directory in the shape RunContext + the benchmark runner produce."""
    logs = root / "logs"
    _write(root / "summary.json", {
        "model": "qwen3.5-2b", "dataset": "bbh_word_sorting", "modality": "llm",
        "n_cases": 6, "baseline_accuracy": 0.436, "cycles": 1, "stopped_by": "max_cycles",
    })
    _write(logs / "manifest.json", {"run_id": "logs", "config": {"fix_tier": "L2", "m1_selection": "pinned"}})
    _write(logs / "report" / "discovery_cases.json", [
        {"id": cid, "prompt": f"sort {cid}", "expected": "a b", "observed": "b a" if label == "fail" else "a b",
         "label": label}
        for cid, label, _ in EXPLORE
    ] + [{"id": "c-5", "prompt": "sort c-5", "expected": "a b", "observed": "a b", "label": "pass"}])
    # one candidate signal, one field derived from the answer key, one constant
    _write(logs / "artifacts" / "c0_coverage_verification_gap.result.json", {
        "findings": {"per_case": [
            {"sample_id": cid, "n_unique": value, "n_correct": 0 if label == "fail" else 1, "n_samples": 5}
            for cid, label, value in EXPLORE
        ]},
    })
    stats = [
        {"tool": "signal_label_assoc", "config": {"signal": SIGNAL}, "effect": 0.65,
         "ci": [0.46, 0.82], "p_value": 2.4e-07, "fdr_corrected": True,
         "correction_method": "BH", "correction_family": "m2_assoc", "summary": "assoc CI=+0.4600..+0.8200"},
        {"tool": "signal_label_assoc", "config": {"signal": "calibration.conf_logprob"}, "effect": 0.05,
         "p_value": 0.78, "fdr_corrected": False, "correction_method": "BH",
         "correction_family": "m2_assoc", "summary": "no association"},
        {"tool": "signal_label_assoc", "config": {"signal": "termination_audit.gave_up"}, "effect": None,
         "fdr_corrected": False, "correction_family": "m2_assoc", "summary": "degenerate: one level only"},
    ]
    _write(logs / "artifacts" / "c0_m2_stats_results.json", stats)
    _write(logs / "artifacts" / "post_m2_stats_results.json", stats)
    events = [
        {"event": "run_start", "judge": "claude", "label_distribution": {"fail": 2, "pass": 2}},
        {"event": "probe", "cycle": 0,
         "selected_analyzers": ["coverage_verification_gap", "logprob_entropy"]},
    ]
    events += [
        {"event": "case_record", "case_id": cid,
         "case": {"label": label, "inputs": {"prompt": f"sort {cid}"}}, "media_paths": []}
        for cid, label, _ in EXPLORE
    ]
    events += [
        {"event": "diagnosis", "hypotheses": [
            {"failure_mode": "`list_integrity_drift", "statement": "words get dropped while re-copying"},
            {"failure_mode": "self_correction_failure", "statement": "re-verify passes corrupt the answer"},
        ]},
        {"event": "surgery", "module": "m5", "failure_mode": "`list_integrity_drift",
         "status": verdict_status, "hypothesis": "words get dropped while re-copying",
         "evidence": {"m5_test_name": "signal_label_assoc", "effect_size": 0.65, "ci": [0.46, 0.82],
                      "m5_evidence_grade": "observational"}},
        {"event": "fix",
         "max_tier": "L2",
         "selection_attempted": [
             {"name": "bucket_then_commit", "tier": "L1", "effect": -0.048},
             {"name": "single_pass_no_revision", "tier": "L1", "effect": -0.2},
             {"name": "self_consistency_5", "tier": "L2", "effect": 0.088},
             {"name": "coded_pipeline", "tier": "L2", "effect": -0.026},
         ],
         "attempted": [{"name": "self_consistency_5", "tier": "L2", "effect": 0.153, "fixed": True}],
         "best": {"name": "self_consistency_5", "tier": "L2", "n_pairs": 124,
                  "n_baseline_correct": 49, "n_candidate_correct": 70,
                  "baseline_rate": 0.4113, "candidate_rate": 0.5645,
                  "n_fixed": 23, "n_broken": n_broken, "effect": 0.1532,
                  "summary": "paired CI=+0.0914..+0.2151", "verdict": "fixed",
                  "payload": {"name": "self_consistency_5", "strategy": "direct", "n_samples": 5,
                              "generation_kwargs": {"temperature": 0.7}, "prompt_template": "{prompt}"}}},
        {"event": "loop_end", "total_duration_sec": 10105.8},
    ]
    (logs / "run_log.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return root


def _events(root: Path) -> list[dict]:
    path = root / "logs" / "run_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def sheet(tmp_path):
    root = make_run(tmp_path / "bbh_word_sorting.claude")
    return build_case_study(root, _events(root))


def test_the_sheet_reads_the_run_from_either_level(tmp_path):
    """An archive is zipped from the run dir or from logs/ — both are the run."""
    root = make_run(tmp_path / "run")
    events = _events(root)
    from_run = build_case_study(root, events)
    from_logs = build_case_study(root / "logs", events)
    assert from_run == from_logs
    assert from_run["headline"]["dataset"] == "bbh_word_sorting"


def test_a_run_with_no_probes_has_no_sheet(tmp_path):
    """Better no section than a sheet of zeroes for a run that never probed."""
    root = tmp_path / "empty"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "run_log.jsonl").write_text("", encoding="utf-8")
    assert build_case_study(root, []) is None


def test_the_funnel_excludes_measurements_computed_from_the_answer_key(sheet):
    m1 = sheet["m1"]
    assert m1["n_analyzers"] == 2
    # n_unique (candidate), n_correct (answer key), n_samples (never varies)
    assert m1["n_measured"] == 3
    assert m1["dropped"] == {"saw_the_answer_key": 1, "never_varied": 1, "partial_coverage": 0}
    asked = [item["question"] for item in m1["questions"]]
    assert asked == ["Same question five times, same answer?"], (
        "the answer-key field is not a question the probe asked")
    # the funnel's far end is M2's correction family: the BH denominator
    assert m1["n_forwarded"] == sheet["m2"]["explore"]["n_in_correction_family"] == 2


def test_the_signal_curve_bins_by_level_with_the_right_denominators(sheet):
    curve = sheet["m1"]["signal_curve"]
    assert curve["signal"] == SIGNAL
    assert curve["binning"] == "levels"
    assert curve["n_cases"] == 4, "only the explore split has labels to bin"
    rates = {bin_["value"]: bin_["failure_rate"] for bin_ in curve["bins"]}
    assert rates == {1: 0.0, 2: 0.0, 5: 1.0}


def test_the_degenerate_test_is_shown_but_kept_out_of_the_correction_family(sheet):
    explore = sheet["m2"]["explore"]
    assert explore["n_in_correction_family"] == 2
    assert explore["n_degenerate"] == 1
    assert explore["survivors"] == [SIGNAL]
    degenerate = [test for test in explore["tests"] if test["degenerate"]]
    assert degenerate and not degenerate[0]["in_correction_family"]


def test_the_ladder_reports_each_tier_from_the_explore_search(sheet):
    ladder = {rung["tier"]: rung for rung in sheet["m4"]["ladder"]}
    assert ladder["L1"]["status"] == "regressed" and ladder["L1"]["n_candidates"] == 2
    assert ladder["L2"]["status"] == "accepted" and ladder["L2"]["best_effect"] == 0.088
    assert ladder["L3"]["status"] == "untouched" and ladder["L3"]["within_cap"] is False
    # the confirmation list is the held-out re-run, not the explore search
    assert [c["name"] for c in sheet["m4"]["confirmed"]] == ["self_consistency_5"]
    assert len(sheet["m4"]["candidates"]) == 4


def test_the_repair_describes_itself_from_its_payload(sheet):
    repair = sheet["repair"]
    assert repair["name"] == "self_consistency_5" and repair["tier"] == "L2"
    titles = [step["title"] for step in repair["steps"]]
    assert titles == ["Prompt", "Sample ×5, take the mode"]


def test_the_paired_validation_adds_up(sheet):
    validation = sheet["validation"]
    assert validation["n_pairs"] == 124
    assert (validation["both_correct"] + validation["both_wrong"]
            + validation["n_fixed"] + validation["n_broken"]) == validation["n_pairs"]
    assert validation["ci"] == [0.0914, 0.2151], "recovered from the summary string"
    assert sheet["headline"]["delta"] == pytest.approx(0.1532, abs=1e-4)


def test_the_caveats_the_sheet_must_carry(sheet):
    codes = {flag["code"] for flag in sheet["qa_flags"]}
    assert "accepted_fix_breaks_cases" in codes
    assert "example_case_from_explore" in codes
    # one survivor only, so the sheet must not warn about plotting one of many
    assert "multiple_signals_survived" not in codes


def test_a_repair_with_no_supported_hypothesis_is_flagged(tmp_path):
    root = make_run(tmp_path / "unsupported", verdict_status="inconclusive", n_broken=0)
    sheet = build_case_study(root, _events(root))
    codes = {flag["code"] for flag in sheet["qa_flags"]}
    assert "repair_without_supported_hypothesis" in codes
    assert "accepted_fix_breaks_cases" not in codes


def test_the_report_payload_carries_the_sheet(tmp_path):
    """build_report_data is where the served UI picks the sheet up."""
    from evalrx.reporting.dynamic import build_report_data, fallback_spec, validate_spec

    root = make_run(tmp_path / "payload")
    data = build_report_data(root / "logs")
    assert data["case_study"]["headline"]["model"] == "qwen3.5-2b"
    spec = validate_spec(fallback_spec(data), data=data)
    types = [element["type"] for element in spec["elements"].values()]
    assert "CaseStudySheet" in types, "the deterministic layout always renders the sheet"
    children = spec["elements"]["page"]["children"]
    assert children.index("case_study") == children.index("journey") + 1


def test_the_sheet_and_the_cli_tool_agree_on_the_numbers(tmp_path):
    """The two readers of these artifacts must not drift apart.

    ``examples/benchmark/tools/extract_figure_data.py`` draws the printed figure
    and this module feeds the served sheet. They are separate code by necessity
    (the tool is stdlib-only and imports nothing from the package), so the risk
    is a silent divergence: the same run described two ways, each plausible.
    """
    import importlib.util

    tool_path = Path(__file__).resolve().parents[2] / "examples" / "benchmark" / "tools" / "extract_figure_data.py"
    if not tool_path.exists():
        pytest.skip("the CLI figure tool is not present in this checkout")
    spec = importlib.util.spec_from_file_location("extract_figure_data", tool_path)
    tool = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tool)

    root = make_run(tmp_path / "agree")
    sheet = build_case_study(root, _events(root))
    document = tool.to_document(tool.extract(str(root)))

    probe = document["m1_probe"]["probe_questions"]
    assert (probe["n_measured"], probe["n_forwarded"]) == (sheet["m1"]["n_measured"], sheet["m1"]["n_forwarded"])
    assert [q["question"] for q in probe["questions"]] == [q["question"] for q in sheet["m1"]["questions"]]
    assert document["m1_probe"]["signal_curve"]["signal"] == sheet["m1"]["signal_curve"]["signal"]
    assert [b["failure_rate"] for b in document["m1_probe"]["signal_curve"]["bins"]] == \
        [b["failure_rate"] for b in sheet["m1"]["signal_curve"]["bins"]]
    assert document["m2_statistics"]["explore"]["family"]["n_in_correction_family"] == \
        sheet["m2"]["explore"]["n_in_correction_family"]
    assert document["heldout_validation"]["n_fixed"] == sheet["validation"]["n_fixed"]
    tool_flags = document["qa_flags"]
    if isinstance(tool_flags, dict):          # the tool nests them under "flags"
        tool_flags = tool_flags.get("flags", [])
    assert {f["code"] for f in tool_flags} == {f["code"] for f in sheet["qa_flags"]}
