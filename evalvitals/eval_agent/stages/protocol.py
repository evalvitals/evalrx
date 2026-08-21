"""ExperimentProtocol — user-supplied description of what an experiment tests.

The protocol is the human prior that anchors the self-evolving loop:

- **M1** passes the protocol to :class:`~evalvitals.eval_agent.probe_agent.ProbeAgent`,
  which uses an LLM judge to select analyzers from the description.
- **M2** uses the protocol to frame its statistical narrative.
- **M5** uses it to verify that a hypothesis is consistent with what the user
  actually set out to investigate.

The description should be written in plain researcher language describing the
*task* and *observed behaviour*.  No failure-mode tags or internal jargon are
needed — the judge LLM interprets the text and selects relevant analyzers.

Usage::

    protocol = ExperimentProtocol(
        description=(
            "We test QwenVL on spatial reasoning. Given an image with two "
            "objects, the model frequently gives wrong left/right and "
            "above/below positions, and sometimes names objects not visible "
            "in the image at all."
        ),
        task_domain="spatial reasoning",
        success_criteria="Positions and object names must match what is visible.",
        target_modalities=frozenset({"text", "image"}),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentProtocol:
    """Natural-language description of an evaluation experiment.

    This is the *human prior* passed into the loop so it can make
    informed, targeted decisions rather than running every analyzer
    blindly.

    Attributes:
        description:        What the experiment tests — free text (required).
        task_domain:        Short label, e.g. ``"spatial reasoning"``,
                            ``"GUI navigation"``.
        success_criteria:   What counts as a pass (used by M5 verifier).
        failure_patterns:   Optional free-text observations about what the
                            researcher has already noticed — passed verbatim
                            to the LLM judge as additional context.
        target_modalities:  ``{"text", "image"}`` for VLMs;
                            ``{"text"}`` for text-only LLMs.
        output_contract:    Optional machine-readable response contract.  It
                            prevents a valid short answer from being confused
                            with a truncated reasoning trace.
        metadata:           Free-form extras (dataset names, hyperparams …).
    """

    description: str
    task_domain: str = ""
    success_criteria: str = ""
    failure_patterns: str = ""
    target_modalities: frozenset[str] = field(
        default_factory=lambda: frozenset({"text"})
    )
    output_contract: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "task_domain": self.task_domain,
            "success_criteria": self.success_criteria,
            "failure_patterns": self.failure_patterns,
            "target_modalities": sorted(self.target_modalities),
            "output_contract": self.output_contract,
            "metadata": self.metadata,
        }


@dataclass
class ProbingSchema:
    """What M1 decided to probe and why.

    Returned by :meth:`~evalvitals.eval_agent.probe_agent.ProbeAgent.probe_with_schema`
    alongside the raw ``{analyzer: Result}`` dict so callers can understand
    *why* those analyzers were chosen.

    Attributes:
        selected_analyzers: Analyzers to run, in priority order.
        rationale:          NL explanation of the selection.
        custom_params:      Per-analyzer parameter overrides (e.g.
                            ``{"attention": {"layer": -1}}``).
        protocol:           The protocol that shaped this schema, if any.
    """

    selected_analyzers: list[str]
    rationale: str = ""
    custom_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    protocol: ExperimentProtocol | None = None
