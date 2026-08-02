"""Agent failure analyzers — operate on a FailureCase's Trajectory.

Build order (cheapest/most reliable first): deterministic heuristics
(loop_detect, ignored_obs) → LLM-judge (first_error_judge, trajectory_rubric)
→ intervention probes that RE-RUN the agent (counterfactual = perturb a step,
reliability_probe = pass@k over k reps, tool_shap = Shapley over tool subsets).
"""

from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay
from evalvitals.analyzers.agent.first_error_judge import FirstErrorJudge
from evalvitals.analyzers.agent.ignored_obs import IgnoredObservationDetector
from evalvitals.analyzers.agent.loop_detect import LoopDetector
from evalvitals.analyzers.agent.reliability import ReliabilityProbe
from evalvitals.analyzers.agent.tool_shap import ToolShap
from evalvitals.analyzers.agent.trajectory_rubric import TrajectoryRubricJudge

__all__ = [
    "LoopDetector",
    "IgnoredObservationDetector",
    "FirstErrorJudge",
    "TrajectoryRubricJudge",
    "CounterfactualReplay",
    "ReliabilityProbe",
    "ToolShap",
]
