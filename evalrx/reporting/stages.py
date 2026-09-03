"""Stage semantics for EvalRX diagnostic reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    id: str
    name: str
    question: str
    artifacts: str
    dashboard_role: str


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        id="M1",
        name="Measurement",
        question="What per-case signals did the evaluators/analyzers extract?",
        artifacts="Frozen per-case feature matrix, analyzer outputs, attention/probe artifacts.",
        dashboard_role="Problem Setting: defines the dataset and available signals.",
    ),
    StageSpec(
        id="M2",
        name="Exploratory analysis",
        question="Which signals are worth investigating, and how do they relate to the outcome?",
        artifacts="Effect sizes, relationship charts/tables — descriptive, no validity verdict.",
        dashboard_role="Analysis: method, evidence, chart, and takeaway (no supported/not-supported "
                        "claim; a confirm phase with e-BH/FDR is a separate, currently out-of-scope step).",
    ),
    StageSpec(
        id="M3",
        name="Hypothesis generation",
        question="What falsifiable failure mechanisms explain the confirmed signals?",
        artifacts="Hypotheses, failure modes, cited M2 charts/observations.",
        dashboard_role="Hypotheses: candidate mechanisms linked back to evidence.",
    ),
    StageSpec(
        id="M4",
        name="Intervention & repair",
        question="Does a targeted intervention repair failures without unacceptable regressions?",
        artifacts="Intervention records, repair candidates, paired outcome comparisons.",
        dashboard_role="Intervene & repair: causal experiments and the repair sweep after validation.",
    ),
    StageSpec(
        id="M5",
        name="Hypothesis validation",
        question="Does corrected statistical evidence and protocol consistency support the hypothesis?",
        artifacts="Held-out verdicts, adjudication records, and evidence grades.",
        dashboard_role="Validate hypotheses: the gate before intervention or repair.",
    ),
)


def stage_specs_as_dicts() -> list[dict[str, str]]:
    return [spec.__dict__.copy() for spec in STAGE_SPECS]
