"""Trajectory rubric judge — failure-mode code + per-dimension rubric scores.

Where :class:`FirstErrorJudge` localises WHERE the failure entered, this judge
classifies WHAT KIND of failure it was (a compact MAST-inspired taxonomy for
single-agent tool loops) and scores the run on five 0-2 rubric dimensions.
The mode is a categorical M2 column; the rubric scores are numeric ones.

Taxonomy (single-agent visual tool loops):
    FM-TOOL-SELECT   needed tool not called / wrong tool for the need
    FM-TOOL-ARGS     malformed or misaimed arguments (wrong region, bad query)
    FM-IGNORE-OBS    answer ignores or contradicts tool evidence
    FM-LOOP          repeats actions without progress
    FM-PERCEPTION    misreads the visual evidence it did gather
    FM-REASONING     right evidence, wrong inference
    FM-ANSWER        correct reasoning, malformed/incomplete final answer
    FM-NONE          no failure visible in the trajectory

References:
- MAST failure-mode taxonomy — Cemri et al., 2025 — arXiv:2503.13657
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Optional

from evalvitals.analyzers.agent.first_error_judge import _render
from evalvitals.core.analyzer import Analyzer
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

FAILURE_MODES = (
    "FM-TOOL-SELECT",
    "FM-TOOL-ARGS",
    "FM-IGNORE-OBS",
    "FM-LOOP",
    "FM-PERCEPTION",
    "FM-REASONING",
    "FM-ANSWER",
    "FM-NONE",
)

RUBRIC_DIMS = ("grounding", "tool_choice", "tool_args", "evidence_use", "answer_quality")

_PROMPT = """You are auditing one AI agent trajectory (a vision-language model calling tools).
Goal: {goal}
Expected answer (may be empty): {expected}
Outcome label (may be unknown): {label}

Steps:
{steps}

Classify the PRIMARY failure mode, one of:
FM-TOOL-SELECT (needed tool not called / wrong tool), FM-TOOL-ARGS (bad or misaimed arguments),
FM-IGNORE-OBS (answer ignores tool evidence), FM-LOOP (repeats without progress),
FM-PERCEPTION (misreads visual evidence), FM-REASONING (right evidence, wrong inference),
FM-ANSWER (malformed/incomplete final answer), FM-NONE (no failure visible).

Then score each rubric dimension 0 (clearly bad), 1 (mixed), or 2 (clearly good):
grounding, tool_choice, tool_args, evidence_use, answer_quality.

Reply with ONLY a JSON object:
{{"failure_mode": "...", "rubric": {{"grounding": 0, "tool_choice": 0, "tool_args": 0, "evidence_use": 0, "answer_quality": 0}}, "first_error_step": <index or -1>, "rationale": "<one sentence>"}}"""


def _parse(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


@register_analyzer("trajectory_rubric")
class TrajectoryRubricJudge(Analyzer):
    """LLM-judged failure-mode classification + rubric scoring per trajectory.

    Hyper-parameters:
        judge:      any object with ``generate(prompt) -> str`` — REQUIRED.
        max_cases:  label-balanced cap on judged cases (one call per case).
        annotate:   when True (default), a valid ``first_error_step`` stamps
                    ``failure_mode`` onto that trajectory step in place.
    """

    name = "trajectory_rubric"
    requires = frozenset()  # the judge is injected, not the probed model
    #: Reads agent runs. The model is not what makes this applicable —
    #: the DATA is, which is why `requires` stays empty (these run on
    #: trajectories loaded from disk, with no model at all).
    requires_trajectories = True
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(self, judge: Any, max_cases: Optional[int] = None, annotate: bool = True) -> None:
        super().__init__(max_cases=max_cases, annotate=annotate)
        if judge is None or not hasattr(judge, "generate"):
            raise ValueError("trajectory_rubric needs a judge with .generate(prompt) -> str")
        self.judge = judge  # ctor name, so sklearn-style get_params() reflection works

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        selected = cases.stratified_head(self.max_cases) if self.max_cases else list(cases)
        per_case: list[dict[str, Any]] = []
        mode_counts: dict[str, int] = {}

        for case in selected:
            traj = case.trajectory
            if traj is None:
                continue
            prompt = _PROMPT.format(
                goal=traj.goal,
                expected=str(case.expected or ""),
                label=case.label.value,
                steps=_render(traj),
            )
            try:
                raw = str(self.judge.generate(prompt))
            except Exception as exc:  # judge outage is a finding, not a crash
                per_case.append({"sample_id": traj.sample_id, "judge_ok": 0, "judge_error": repr(exc)})
                continue
            payload = _parse(raw)
            if not payload:
                per_case.append(
                    {"sample_id": traj.sample_id, "judge_ok": 0, "judge_raw": raw[:300]}
                )
                continue

            mode = str(payload.get("failure_mode", "")).upper()
            if mode not in FAILURE_MODES:
                mode = "FM-NONE" if case.label.value == "pass" else "FM-REASONING"
            entry: dict[str, Any] = {
                "sample_id": traj.sample_id,
                "judge_ok": 1,
                "failure_mode": mode,
                "rationale": str(payload.get("rationale", ""))[:300],
            }
            rubric = payload.get("rubric") or {}
            for dim in RUBRIC_DIMS:
                val = rubric.get(dim)
                if isinstance(val, (int, float)):
                    entry[f"rubric_{dim}"] = max(0, min(2, int(val)))
            step_idx = payload.get("first_error_step")
            if isinstance(step_idx, int) and 0 <= step_idx < len(traj.steps):
                entry["first_error_step"] = step_idx
                if self.annotate:
                    traj.steps[step_idx].failure_mode = mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            per_case.append(entry)

        findings: dict[str, Any] = {
            "n_trajectories": len(per_case),
            "n_judged": sum(c.get("judge_ok", 0) for c in per_case),
            "judge": repr(self.judge),
            "mode_counts": mode_counts,
            "per_case": per_case,
            "_caveat": (
                "Judge-derived labels — treat as hypotheses to verify, not ground "
                "truth. failure_mode is categorical (records-path column); "
                "rubric_* are 0-2 numerics."
            ),
        }
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
