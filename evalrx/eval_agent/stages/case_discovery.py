"""Case discovery and labeling for automated VL diagnosis loops.

This stage turns candidate prompts into labeled cases by running the target
model, storing its observed answer, and scoring the result.  It is intentionally
small: dataset loading and prompt generation can live outside this class, while
M4 receives the PASS/FAIL labels it needs for statistical testing.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal

from pydantic import BaseModel, field_validator

from evalrx.agent_runtime.judge_call import JudgeOutputError, ask_judge
from evalrx.core.case import CaseBatch, FailureCase, Label
from evalrx.eval_agent.prompts.case_discovery import _JUDGE_PROMPT

if TYPE_CHECKING:
    from evalrx.core.model import Judge, Model
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol


ScoreFn = Callable[[FailureCase, str], Label | bool | str]


@dataclass
class CaseDiscoveryReport:
    """Summary of one discovery pass."""

    cases: CaseBatch
    n_pass: int = 0
    n_fail: int = 0
    n_unknown: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_m4_groups(self) -> bool:
        """True when M4 has both failure and control examples."""
        return self.n_fail > 0 and self.n_pass > 0


class CaseDiscoveryAgent:
    """Run candidate cases through a model and label PASS/FAIL outcomes.

    Args:
        scorer:            Deterministic scoring function.  It receives the
                           case and observed answer and returns ``Label``, bool,
                           or a label string.
        judge:             Optional LLM judge used when no scorer is supplied.
        generation_kwargs: Extra kwargs forwarded to ``model.generate``.
        include_unknown:   Keep UNKNOWN cases in the returned batch.
        n_samples:         Answers drawn per case; the label is their majority.
                           One is right for a deterministic model. A sampling
                           model can answer the same case both ways, so one
                           answer labels it by chance; use an odd number.
    """

    def __init__(
        self,
        scorer: ScoreFn | None = None,
        judge: "Judge | None" = None,
        generation_kwargs: dict[str, Any] | None = None,
        include_unknown: bool = True,
        n_samples: int = 1,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        self.scorer = scorer
        self.judge = judge
        self.generation_kwargs = generation_kwargs or {}
        self.include_unknown = include_unknown
        self.n_samples = n_samples

    def discover(
        self,
        model: "Model",
        candidates: Iterable[FailureCase] | CaseBatch,
        *,
        protocol: "ExperimentProtocol | None" = None,
    ) -> CaseDiscoveryReport:
        """Return a labeled batch created from candidate cases."""
        labeled: list[FailureCase] = []
        errors: list[str] = []
        counts = {Label.PASS: 0, Label.FAIL: 0, Label.UNKNOWN: 0}
        cases = list(candidates)

        for case, answers in zip(cases, self._generate(model, cases)):
            samples = [self._sample(case, answer, protocol) for answer in answers]
            n_pass = sum(1 for s in samples if s.label is Label.PASS)
            n_fail = sum(1 for s in samples if s.label is Label.FAIL)
            label = (Label.PASS if n_pass > n_fail
                     else Label.FAIL if n_fail > n_pass else Label.UNKNOWN)
            # The recorded answer is one that earned the label: analyzers score
            # ``observed`` and must reach the label the case carries.
            shown = next((s for s in samples if s.label is label), samples[0])
            failed = [s.error for s in samples if s.error]
            errors.extend(f"{case.id}: {error}" for error in failed)

            if shown.observed is not None:
                case.observed = shown.observed
            case.label = label
            case.metadata["discovery_observed"] = "" if shown.error else str(shown.observed)
            case.metadata["discovery_label"] = label.value
            case.metadata["discovery_reason"] = shown.reason
            if label is Label.UNKNOWN and failed:
                case.metadata["discovery_error"] = failed[0]
            if self.n_samples > 1:
                case.metadata["discovery_votes"] = {"pass": n_pass, "fail": n_fail}
                first = next((s for s in samples if not s.error), shown)
                if first is not shown:
                    case.metadata[_FIRST_ANSWER] = str(first.observed)
                case.metadata["discovery_reason"] = (
                    f"{n_pass} of {len(samples)} sampled answers passed; {shown.reason}")
            counts[label] += 1

            if label != Label.UNKNOWN or self.include_unknown:
                labeled.append(case)

        return CaseDiscoveryReport(
            cases=CaseBatch(labeled),
            n_pass=counts[Label.PASS],
            n_fail=counts[Label.FAIL],
            n_unknown=counts[Label.UNKNOWN],
            errors=errors,
        )

    def _generate(self, model: "Model", cases: list[FailureCase]) -> "list[list[Any]]":
        """Each case's ``n_samples`` answers, in case order.

        A failed generation is carried as its exception rather than raised, so
        one bad case leaves the batch intact.
        """
        # One batch call. The backend decides how many run at once: padded
        # batches on a local model, ``runtime.concurrency`` requests over an API.
        k = self.n_samples
        answers = model.try_generate_batch(
            [case.inputs for case in cases for _ in range(k)], **self.generation_kwargs)
        return [answers[i * k:(i + 1) * k] for i in range(len(cases))]

    def _sample(
        self, case: FailureCase, answer: Any, protocol: "ExperimentProtocol | None"
    ) -> "_Sample":
        """One answer and its label; a failed generation or scoring is UNKNOWN."""
        if isinstance(answer, Exception):
            return _Sample(None, Label.UNKNOWN, _FAILED, str(answer))
        try:
            label, reason = self._score_with_reason(case, str(answer), protocol)
        except Exception as exc:  # noqa: BLE001 - discovery should continue
            return _Sample(answer, Label.UNKNOWN, _FAILED, str(exc))
        return _Sample(answer, label, reason)

    def _score_with_reason(
        self,
        case: FailureCase,
        observed: str,
        protocol: "ExperimentProtocol | None",
    ) -> tuple[Label, str]:
        if self.scorer is not None:
            label = _coerce_label(self.scorer(case, observed))
            return label, "scored by injected scorer"
        if self.judge is not None:
            judged, reason = self._judge_score(case, observed, protocol)
            if judged != Label.UNKNOWN:
                return judged, reason
            fallback = _heuristic_score(case.expected, observed)
            return fallback, f"judge returned UNKNOWN; heuristic fallback produced {fallback.value}"
        label = _heuristic_score(case.expected, observed)
        return label, "scored by expected-answer heuristic"

    def _judge_score(
        self,
        case: FailureCase,
        observed: str,
        protocol: "ExperimentProtocol | None",
    ) -> tuple[Label, str]:
        protocol_text = protocol.description if protocol is not None else ""
        success_criteria = protocol.success_criteria if protocol is not None else ""
        prompt = _JUDGE_PROMPT.format(
            protocol=protocol_text,
            success_criteria=success_criteria,
            prompt=case.inputs.prompt,
            expected=case.expected,
            observed=observed,
        )
        try:
            graded = ask_judge(self.judge, prompt, _CaseLabel, role="case_label").output
        except JudgeOutputError as exc:
            return Label.UNKNOWN, f"judge gave no valid label ({exc})"
        return _coerce_label(graded.label), graded.reason


_FAILED = "model generation or scoring raised an exception"
_FIRST_ANSWER = "discovery_first_answer"


def recorded_draw(case: Any) -> "str | None":
    """One answer of the unmodified model to *case*, drawn without regard to its label.

    That is ``case.observed``, except under a majority label: discovery then
    picks ``observed`` to agree with the label, so it is no longer a fair
    sample, and keeps the first answer it drew beside it. A comparison against
    the unmodified model (a baseline rate, a fallback, a replay) reads this.
    """
    first = (getattr(case, "metadata", None) or {}).get(_FIRST_ANSWER)
    if first is not None:
        return str(first)
    observed = getattr(case, "observed", None)
    return None if observed is None else str(observed)


@dataclass
class _Sample:
    """One sampled answer of a case and what it scored."""

    observed: Any
    label: Label
    reason: str
    error: str = ""


class _CaseLabel(BaseModel):
    label: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str = "judge returned a structured label"

    @field_validator("label", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


def _coerce_label(value: Label | bool | str) -> Label:
    if isinstance(value, Label):
        return value
    if isinstance(value, bool):
        return Label.PASS if value else Label.FAIL
    text = str(value).strip().lower()
    if text in {"pass", "passed", "true", "correct", "ok"}:
        return Label.PASS
    if text in {"fail", "failed", "false", "incorrect", "wrong"}:
        return Label.FAIL
    return Label.UNKNOWN


def _heuristic_score(expected: Any, observed: str) -> Label:
    """Score simple expected-answer rubrics without an LLM judge.

    Supported expected formats:
      - string: substring match after normalization
      - list/tuple/set: every item must appear
      - dict: ``all_of``, ``any_of``, and ``none_of`` substring constraints
    """
    if expected is None:
        return Label.UNKNOWN

    obs = _normalize(observed)
    if isinstance(expected, str):
        return Label.PASS if _normalize(expected) in obs else Label.FAIL

    if isinstance(expected, (list, tuple, set)):
        terms = [_normalize(x) for x in expected]
        return Label.PASS if all(t in obs for t in terms if t) else Label.FAIL

    if isinstance(expected, dict):
        all_of = [_normalize(x) for x in expected.get("all_of", [])]
        any_of = [_normalize(x) for x in expected.get("any_of", [])]
        none_of = [_normalize(x) for x in expected.get("none_of", [])]
        if any(t and t in obs for t in none_of):
            return Label.FAIL
        if all_of and not all(t in obs for t in all_of if t):
            return Label.FAIL
        if any_of and not any(t in obs for t in any_of if t):
            return Label.FAIL
        return Label.PASS

    return Label.UNKNOWN


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()
