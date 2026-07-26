"""Prompt addenda for Agent Skill usage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalvitals.agent_runtime.cli_types import CliAgentConfig


def fences_hint(cli_config: "CliAgentConfig | None") -> str:
    if cli_config is not None and cli_config.provider != "llm":
        return ", written to a file named analysis.py"
    return " inside a ```python code block"


def skills_hint(cli_config: "CliAgentConfig | None") -> str:
    """Prompt addendum steering the agent to apply available Agent Skills."""
    if cli_config is None or not getattr(cli_config, "skills_enabled", False):
        return ""

    names = [Path(path).name for path in (cli_config.skills or [])]
    is_codex = getattr(cli_config, "provider", "") == "codex"

    def _use(nlist: list[str]) -> str:
        if not nlist:
            if is_codex:
                return (
                    "read the vendored guides under `.claude/skills/` "
                    "(also listed in AGENTS.md) and apply them"
                )
            return "invoke any installed Agent Skills via the Skill tool and follow them"
        if is_codex:
            refs = ", ".join(f"`.claude/skills/{name}/SKILL.md`" for name in nlist)
            return f"read the vendored guides ({refs} -- also listed in AGENTS.md) and apply them"
        listed = ", ".join(f"`/{name}`" for name in nlist)
        return f"invoke the {listed} skill(s) via the Skill tool and follow them"

    parts: list[str] = []

    if "outcome-driver-analysis" in names:
        parts.append(
            "ANALYSIS METHOD: BEFORE writing any analysis code, "
            + _use(["outcome-driver-analysis"])
            + " to choose justified statistical methods for the outcome analysis: "
            "explanatory-variable EDA, per-variable tests WITH effect sizes, "
            "conditioning/confounding (Simpson's) checks, marginal screening, a "
            "justified regression model (state the GLM-vs-mixed-effects reasoning "
            "from the ACTUAL clustering structure -- with very few clusters prefer "
            "a fixed effect over a random effect), and fit diagnostics "
            "(collinearity/VIF, discrimination, calibration). Adopt its "
            "METHODOLOGY, not its file layout or report template -- every output "
            "still flows into the required result-JSON contract, and intake "
            "answers are inferred from the data profile and the question (never "
            "stop to ask). Keep all wording DESCRIPTIVE: takeaways carry effect "
            "sizes + confidence intervals + direction; test statistics/p-values "
            "belong in tables/artifacts and must NOT be phrased as significance "
            "or confirmation verdicts -- validity is adjudicated downstream."
        )

    style = [name for name in names if name != "outcome-driver-analysis"]
    roles = []
    if "eval-chart-style" in style:
        roles.append(
            "`eval-chart-style` governs chart-TYPE choice (distribution-first: "
            "violin/ECDF/heatmap/forest/paired-slope -- never a mean as a bar) and "
            "the palette: FAIL/PASS semantic hues, a categorical series order for "
            "non-outcome dimensions, and single-hue ramps for ordered dimensions "
            "(e.g. model sizes). When any other skill suggests a specific chart "
            "type or color, eval-chart-style's policy and palette take precedence"
        )
    if "nature-figure" in style:
        roles.append("`nature-figure` adds publication-grade matplotlib polish")
    role_line = ("; ".join(roles) + ". ") if roles else ""
    parts.append(
        "FIGURE STYLING: Agent Skills are available -- apply them BY DEFAULT to "
        f"every plot you write under figures/: BEFORE plotting, {_use(style)}. "
        f"{role_line}"
        "This is a non-interactive PYTHON analysis: if a skill asks you to choose "
        "a plotting backend, choose Python and proceed without pausing -- never "
        "stop to ask a question. Use skills for styling only -- they must not "
        "change the data, the analysis, the sandbox workflow, or the final "
        "result JSON."
    )
    return "\n" + "\n\n".join(parts) + "\n"
