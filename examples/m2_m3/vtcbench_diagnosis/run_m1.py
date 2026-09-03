"""VTC-Bench agent diagnosis — M1: batch the agent, probe it, emit records.json.

One VTC-Bench task (default: counting, 85 four-way multiple-choice cases) runs
through the vLLM-served Qwen3-VL agent with {image_zoom_in, image_detect}:

  stage 1  base batch                      -> trajectories + graded labels
  stage 2  loop_detect / ignored_obs      (all cases, free)
  stage 3  reliability_probe  k=3 @ t=0.7 (stratified subset — interventional)
  stage 4  tool_shap 2^2 subsets, exact   (same subset — interventional)
  stage 5  trajectory_rubric              (same subset; served model as judge)
  stage 6  merge everything into outputs/records.json for `evalrx explore`

Usage:
    .venv/bin/python examples/m2_m3/vtcbench_diagnosis/run_m1.py \
        --task counting --base-url http://localhost:8901/v1 --probe-cases 24
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from evalrx import compose
from evalrx.analysis.trajectory_records import trajectories_to_records
from evalrx.analyzers.agent.ignored_obs import IgnoredObservationDetector
from evalrx.analyzers.agent.loop_detect import LoopDetector
from evalrx.analyzers.agent.reliability import ReliabilityProbe
from evalrx.analyzers.agent.tool_shap import ToolShap
from evalrx.analyzers.agent.trajectory_rubric import TrajectoryRubricJudge
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.models.agent import Agent, run_batch
from evalrx.models.backends.openai_compat import openai_runtime
from evalrx.models.tools import detect_tool, zoom_in_tool
from evalrx.models.tools.perception import default_detect_engine

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/tealab-data/jiaqiliu/datasets"
QA_PATH = os.path.join(DATA_ROOT, "vtcbench", "VTC-Bench_qa.json")

SYSTEM = (
    "You are a careful visual reasoning agent. Use the available tools to "
    "inspect the image before answering: image_zoom_in enlarges a region, "
    "image_detect finds and boxes named objects. Zoom into relevant regions "
    "when detail matters. For multiple-choice questions, END your final "
    "answer with the single option letter (A, B, C, or D) and nothing after "
    "it. Give the final answer as plain text without any tool call."
)

_LETTER = re.compile(r"\b([A-D])\b")


def mc_grader(trajectory, case):
    """Grade a four-way MC answer: LAST standalone A-D letter vs the solution's.

    A run that never produced a final answer (loop exhausted max_turns) IS a
    failure, not an ungradable case — otherwise one non-answering subset run
    would void a whole tool_shap/reliability entry.
    """
    expected = str(case.expected or "").strip()
    if not expected:
        return None
    if trajectory.final_answer is None:
        return False
    hits = _LETTER.findall(str(trajectory.final_answer))
    return bool(hits) and hits[-1] == expected


def load_cases(task: str, limit: int | None) -> list[FailureCase]:
    items = json.load(open(QA_PATH))
    cases: list[FailureCase] = []
    for it in items:
        doc_task = it["doc_id"].replace("vtcbench_", "").rsplit("_", 1)[0]
        if doc_task != task:
            continue
        img_path = os.path.join(DATA_ROOT, it["images"][0])
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        solution = str(it["solution"]).strip()
        letter = solution[0] if solution[:1] in "ABCD" and solution[1:2] in (".", ")", "") else None
        if letter is None:
            continue  # this run grades MC only; free-text items are skipped
        prompt = it["problem"].replace("<image>\n", "").replace("<image>", "").strip()
        cases.append(
            FailureCase(
                inputs=Inputs(prompt=prompt, image=Image.open(img_path).convert("RGB")),
                expected=letter,
                metadata={"doc_id": it["doc_id"], "task": task, "solution": solution},
            )
        )
        if limit and len(cases) >= limit:
            break
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="counting")
    ap.add_argument("--base-url", default="http://localhost:8901/v1")
    ap.add_argument("--model", default="qwen3-vl-2b-instruct")
    ap.add_argument("--limit", type=int, default=None, help="cap total cases (smoke runs)")
    ap.add_argument("--probe-cases", type=int, default=24, help="stratified cap for interventional probes")
    ap.add_argument("--k", type=int, default=3, help="reliability repetitions")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--detect-device", default="cuda:2")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument(
        "--no-loop-policy", action="store_true",
        help="disable the VALIDATED default loop policy (block_repeat_calls + "
             "force_final_answer) to reproduce the unfixed baseline — required "
             "for run_m4.py fix comparisons",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # L2 loop policy is the example's recommended default: validated across six
    # independent VTC batches (combined e ~ 1.6e10; see README / CHANGELOG).
    agent_kwargs = (
        {} if args.no_loop_policy
        else {"block_repeat_calls": True, "force_final_answer": True}
    )

    cases = load_cases(args.task, args.limit)
    policy = "OFF (unfixed baseline)" if args.no_loop_policy else "ON (validated default)"
    print(f"[0] {len(cases)} MC cases for task={args.task!r} | loop policy {policy}")
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"task": args.task, "model": args.model, "max_turns": args.max_turns,
                   "loop_policy": not args.no_loop_policy}, f, indent=1)

    engine = default_detect_engine(device=args.detect_device)
    vlm = compose(args.model, "api",
                  runtime=openai_runtime(base_url=args.base_url, max_tokens=512))
    vlm_t = compose(args.model, "api",
                    runtime=openai_runtime(base_url=args.base_url, max_tokens=512, temperature=0.7))

    def tools_for(case, names=("image_zoom_in", "image_detect")):
        tools = []
        if "image_zoom_in" in names:
            tools.append(zoom_in_tool(case.inputs.image))
        if "image_detect" in names:
            tools.append(detect_tool(case.inputs.image, engine=engine))
        return tools

    # -- stage 1: base batch + grading ---------------------------------
    t0 = time.time()
    trajs = run_batch(vlm, cases, tools_factory=tools_for, system=SYSTEM,
                      max_turns=args.max_turns, concurrency=args.concurrency,
                      agent_kwargs=agent_kwargs)
    n_pass = 0
    for case, traj in zip(cases, trajs):
        case.trajectory = traj
        case.observed = traj.final_answer
        graded = mc_grader(traj, case)
        case.label = Label.PASS if graded else Label.FAIL
        n_pass += bool(graded)
    print(f"[1] base batch: {time.time()-t0:.0f}s | pass {n_pass}/{len(cases)} "
          f"(fail rate {1 - n_pass/len(cases):.2f})")
    batch = CaseBatch(cases)
    with open(os.path.join(args.out, "trajectories.jsonl"), "w") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    # -- stage 2: observational analyzers (all cases) ------------------
    results = {
        "loop_detect": LoopDetector().run(vlm, batch),
        "ignored_obs": IgnoredObservationDetector().run(vlm, batch),
    }

    # -- probe subset (label-stratified) -------------------------------
    probe_cases = batch.stratified_head(args.probe_cases)
    probe_batch = CaseBatch(probe_cases)
    print(f"[2] probes on {len(probe_cases)} stratified cases "
          f"({sum(c.label is Label.FAIL for c in probe_cases)} FAIL)")

    # -- stage 3: reliability (precomputed, concurrent) ----------------
    t0 = time.time()
    reps = [c for c in probe_cases for _ in range(args.k)]
    rep_trajs = run_batch(vlm_t, reps, tools_factory=tools_for, system=SYSTEM,
                          max_turns=args.max_turns, concurrency=args.concurrency,
                          agent_kwargs=agent_kwargs)
    rel_cache: dict[str, list] = {}
    for case, traj in zip(reps, rep_trajs):
        rel_cache.setdefault(case.id, []).append(traj)
    results["reliability_probe"] = ReliabilityProbe(
        runs_fn=lambda case, k: rel_cache.get(case.id, []), k=args.k, grader=mc_grader,
    ).run(vlm, probe_batch)
    print(f"[3] reliability: {time.time()-t0:.0f}s")

    # -- stage 4: tool_shap (precomputed subset runs, concurrent) ------
    t0 = time.time()
    subsets = [(), ("image_detect",), ("image_zoom_in",), ("image_zoom_in", "image_detect")]
    pairs = [(case, names) for case in probe_cases for names in subsets]

    def _run_pair(pair):
        case, names = pair
        try:
            return Agent(vlm, tools_for(case, names), system=SYSTEM,
                         max_turns=args.max_turns, **agent_kwargs).run(case)
        except Exception as exc:  # e.g. context overflow — a failed run, not a dead batch
            from evalrx.core.case import Step, StepRole, Trajectory
            return Trajectory(
                sample_id=case.id, goal=case.inputs.prompt,
                steps=[Step(idx=0, role=StepRole.USER, content=case.inputs.prompt)],
                metrics={"terminated": "error", "error": repr(exc)[:200]},
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pair_trajs = list(pool.map(_run_pair, pairs))
    shap_cache = {(case.id, frozenset(names)): traj
                  for (case, names), traj in zip(pairs, pair_trajs)}
    results["tool_shap"] = ToolShap(
        run_with_tools=lambda case, names: shap_cache[(case.id, frozenset(names))],
        tool_names=["image_zoom_in", "image_detect"], grader=mc_grader,
    ).run(vlm, probe_batch)
    print(f"[4] tool_shap: {time.time()-t0:.0f}s")

    # -- stage 5: rubric (served model as judge; noisy, caveated) ------
    t0 = time.time()
    results["trajectory_rubric"] = TrajectoryRubricJudge(judge=vlm).run(vlm, probe_batch)
    print(f"[5] rubric: {time.time()-t0:.0f}s | "
          f"modes={results['trajectory_rubric'].findings['mode_counts']}")

    # -- stage 6: merge into records.json ------------------------------
    rows = trajectories_to_records(cases)
    by_id: dict[str, dict] = {}
    for name, result in results.items():
        for entry in result.findings.get("per_case", []):
            sid = entry.get("sample_id")
            if not sid:
                continue
            dst = by_id.setdefault(sid, {})
            for key, value in entry.items():
                if key in ("sample_id", "runs", "loops", "ignored", "rationale", "judge_raw"):
                    continue
                if key == "baseline_pass":  # label-duplicative by construction — keep out
                    continue
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float, str)) or value is None:
                    dst[f"{name}__{key}" if key in ("n_runs", "n_graded") else key] = value
    for row in rows:
        row.update(by_id.get(row["case_id"], {}))
    with open(os.path.join(args.out, "records.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "probe_findings.json"), "w") as f:
        json.dump({k: v.findings for k, v in results.items()}, f, ensure_ascii=False, indent=1, default=str)
    n_cols = len(rows[0]) if rows else 0
    print(f"[6] wrote {len(rows)} rows x ~{n_cols} cols -> {args.out}/records.json")


if __name__ == "__main__":
    main()
