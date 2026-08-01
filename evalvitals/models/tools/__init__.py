"""Subject-side tool implementations for the :class:`~evalvitals.models.agent.Agent` loop.

These are the tools an agent-under-test may call — distinct from
``eval_agent.agentic`` (the host's *diagnosis* tools) and ``agent_runtime``
(the analysis-side CLI coding agents).  Design constraints, because these runs
are diagnosed, not just executed:

* **deterministic** — no sampling, versioned weights, so the tool layer stays a
  controlled variable under held-out re-evaluation and fix-vs-baseline pairing;
* **model-agnostic schemas** — normalized ``[0,1]`` coordinates, conventions
  spelled out in the tool description (the description is an L1 fix surface);
* **structured results** — :class:`~evalvitals.core.tool.ToolResult` with
  model-visible text, re-injectable images, and host-side ``meta`` that lands
  on the trajectory step.
"""

from evalvitals.models.tools.visual import zoom_in_tool

__all__ = ["zoom_in_tool"]
