"""Audit completed default-agentic runs; does not make model calls."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
from evalrx.benchmark import tasks as T


def read(path):
    return json.loads(path.read_text())


rows = []
selected = set(sys.argv[1:])
for modality, dataset in [
    ("llm", "bbh_causal_judgement"), ("vlm", "chartqa"), ("alm", "mmau")
]:
    if selected and modality not in selected:
        continue
    run_key = "llm-retry" if modality == "llm" else modality
    name = f"evalrx-agentic-default-20261012-{run_key}"
    state = json.loads(subprocess.check_output(
        ["docker", "inspect", "--format", "{{json .State}}", name], text=True
    ))
    assert state["Status"] == "exited", (name, state["Status"])
    p = ROOT / "runs" / run_key / "gemini-3.6-flash" / dataset
    row = dict(modality=modality, dataset=dataset, container=name,
               status=state["Status"], exit_code=state["ExitCode"], run_dir=str(p))
    (ROOT / f"{modality}.log").write_text(subprocess.check_output(
        ["docker", "logs", name], stderr=subprocess.STDOUT, text=True
    ))
    row["summary"] = read(p / "summary.json")
    baseline = read(p / "baseline.json")
    row["baseline_correct"] = baseline["n_pass"]
    cases = read(p / "logs/run.json")["cases"]
    splits = {s: {x["case_id"] for x in cases if x["split"] == s}
              for s in ["explore", "confirm"]}
    assert not splits["explore"] & splits["confirm"]
    row["split_counts"] = {s: len(v) for s, v in splits.items()}
    row["stage_evidence"] = {}
    for stage in [f"M{i}" for i in range(1, 6)]:
        doc = read(p / "logs" / stage / "log.json")
        row["stage_evidence"][stage] = {k: len(v) for k, v in doc.items()}
    doc = read(p / "logs/M5/log.json")
    row["stage_skipped"] = doc.get("stage_skipped", [])
    row["experiments"] = [{k: x.get(k) for k in
                           ["status", "returncode", "timed_out", "metrics"]}
                          for x in doc.get("experiment", [])]
    fixes = doc.get("fix", [])
    if fixes:
        fix = fixes[-1]
        row["fix"] = {k: fix.get(k) for k in
                      ["fixed", "recommendation", "stage_status", "skip_reason",
                       "selected_on_explore"]}
        row["fix"]["selection_attempts"] = len(fix.get("selection_attempted", []))
        row["fix"]["confirm_attempts"] = []
        for a in fix.get("attempted", []):
            out = {k: a.get(k) for k in [
                "name", "tier", "n_pairs", "n_baseline_correct", "n_candidate_correct",
                "n_fixed", "n_broken", "effect", "verdict", "fixed", "e_value",
                "e_threshold", "noise_model", "n_unstable", "n_baseline_samples",
                "n_candidate_samples"]}
            assert set(a.get("outputs", {})) <= splits["confirm"]
            manifest = REPO / "examples/benchmark" / modality / "_data" / dataset / "manifest.json"
            cs, _ = T.build_cases(T.get(dataset), manifest, 0)
            by_id = {c.id: c for c in cs}
            scores = {cid: bool(T.score_case(by_id[cid], value))
                      for cid, value in a.get("outputs", {}).items()}
            out["rescored_outputs"] = scores
            out["rescored_output_correct"] = sum(scores.values())
            base = {x["case_id"]: x["case"]["label"] == "pass"
                    for x in cases if x["split"] == "confirm"}
            counts = {
                "n_pairs": len(scores),
                "n_baseline_correct": sum(base[k] for k in scores),
                "n_candidate_correct": sum(scores.values()),
                "n_fixed": sum(scores[k] and not base[k] for k in scores),
                "n_broken": sum(base[k] and not scores[k] for k in scores),
            }
            if a.get("n_baseline_samples") == a.get("n_candidate_samples") == 1:
                assert all(a[k] == v for k, v in counts.items()), (a["name"], counts)
            out["rescored_counts"] = counts
            row["fix"]["confirm_attempts"].append(out)
    rows.append(row)
changed = []
for filename, snapshot in [
    ("source_sha256.json", "/tmp/evalrx-agentic-default-20261012"),
    ("source_sha256_llm_retry.json", "/tmp/evalrx-agentic-default-20261012-llm-retry"),
]:
    source_hashes = read(ROOT / filename)
    changed += [f for f, digest in source_hashes.items()
                if hashlib.sha256((Path(snapshot) / f).read_bytes()).hexdigest() != digest]
assert not changed, changed
result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
              source_unchanged=True, runs=rows)
(ROOT / (("_".join(sorted(selected)) + "_audit.json") if selected else "full_chain_audit.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
