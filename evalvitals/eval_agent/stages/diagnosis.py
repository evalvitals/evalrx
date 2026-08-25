"""M3 — DiagnosisAgent: synthesize analysis findings into falsifiable hypotheses.

The agent reads the structured :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
produced by M2, formats it as a prompt, sends it to an LLM judge (Gemini by
default), and parses the response into :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
objects.

The judge model defaults to **Gemini** when ``GEMINI_API_KEY`` is set in the
environment and no explicit judge is passed.

Usage::

    # Gemini default (set GEMINI_API_KEY in env)
    agent = DiagnosisAgent()
    diag  = agent.diagnose(analysis_report)

    # Explicit judge (any Model with Capability.GENERATE)
    agent = DiagnosisAgent(judge=my_model)
    diag  = agent.diagnose(analysis_report)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.agent_runtime.json_shape import validate_json_shape
from evalvitals.analysis.plain_language import jargon_violation
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.prompts.diagnosis import (
    _DIAGNOSE_PROMPT,
    _PLAIN_REPAIR_PROMPT,
    _VALIDATE_PROMPT,
)

if TYPE_CHECKING:
    from evalvitals.analysis.analysis_module import AnalysisReport
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result

def _format_prior_section(prior_cycles: list[dict]) -> str:
    """Render prior cycle history as a context block for the diagnosis prompt."""
    if not prior_cycles:
        return ""
    lines = [
        "\nPrior investigation cycles — do NOT repeat these hypotheses; propose "
        "distinct or refined ones that address what remains unexplained:",
    ]
    for c in prior_cycles:
        lines.append(f"\nCycle {c['cycle']} (severity={c['severity']}):")
        for h in c.get("hypotheses", []):
            lines.append(
                f"  [{h['status'].upper()}] {h['statement']} "
                f"(failure_mode: {h['failure_mode']})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class ExploreContext:
    """Descriptive mechanism notes from the LAMBDA explorer (Step 1).

    This is **never authoritative**: it is read-only, enters ONLY the M3
    hypothesis-proposal prompt (and the dashboard), and never the M2 confirmatory
    family, M5 testing, or the fix gate. It informs *which* hypotheses M3
    proposes — not *whether* any of them is true. Every chart/observation here is
    free-form EDA on a HELD-OUT explore split and stays UNCONFIRMED until the
    downstream M5+M4+e-BH machinery tests it.

    Attributes:
        observations: Free-text EDA observations.
        charts:       Chart dicts ``{title, kind, description, figure_path}`` —
                      ``figure_path`` (when present) is attached to M3 as an image.
        caveats:      The explorer's own warnings about its findings.
        source:       Provenance tag (default ``"lambda_explorer"``).
    """

    observations: list[str] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    source: str = "lambda_explorer"

    @classmethod
    def from_report(cls, data: dict[str, Any] | None) -> "ExploreContext | None":
        """Build from an explore/fused report dict (``fused_report.json`` etc.).

        Returns ``None`` when *data* is empty or carries no descriptive content,
        so callers can pass it through unconditionally.
        """
        if not data or not isinstance(data, dict):
            return None
        observations = [str(x) for x in (data.get("observations") or [])]
        charts = [dict(c) for c in (data.get("charts") or []) if isinstance(c, dict)]
        caveats = [str(x) for x in (data.get("caveats") or [])]
        if not (observations or charts or caveats):
            return None
        return cls(
            observations=observations,
            charts=charts,
            caveats=caveats,
            source=str(data.get("source") or "lambda_explorer"),
        )

    @property
    def figure_paths(self) -> list[str]:
        return [
            str(c["figure_path"])
            for c in self.charts
            if isinstance(c, dict) and c.get("figure_path")
        ]

    @property
    def is_empty(self) -> bool:
        return not (self.observations or self.charts or self.caveats)


def _format_explore_section(ctx: "ExploreContext | None") -> str:
    """Render an :class:`ExploreContext` as a strongly-labelled, UNCONFIRMED
    block for the diagnosis prompt. Returns ``""`` when there is nothing to add."""
    if ctx is None or ctx.is_empty:
        return ""
    lines = [
        "",
        "EXPLORATORY MECHANISM NOTES (free-form EDA on a HELD-OUT explore split — "
        "DESCRIPTIVE, UNCONFIRMED; use ONLY to decide WHICH hypotheses to propose, "
        "NEVER as evidence; every claim must still be tested downstream):",
    ]
    if ctx.observations:
        lines.append("  observations:")
        for obs in ctx.observations[:12]:
            lines.append(f"    - {obs}")
    if ctx.charts:
        lines.append("  charts (attached as images when rendered):")
        for c in ctx.charts[:12]:
            title = str(c.get("title") or c.get("name") or "chart")
            desc = str(c.get("description") or c.get("kind") or "")
            tag = "" if c.get("figure_path") else " [text-only, not rendered]"
            lines.append(f"    - [{title}]{tag} {desc}")
    if ctx.caveats:
        lines.append("  caveats (the explorer's own warnings):")
        for cav in ctx.caveats[:8]:
            lines.append(f"    - {cav}")
    return "\n".join(lines) + "\n"


def _format_failure_modes_section(report: "Any | None") -> str:
    """Render a :class:`~evalvitals.analysis.failure_modes.FailureModeReport`
    as a labelled block for the diagnosis prompt. Descriptive only — clustering
    groups FAIL cases by similarity, it does not itself confirm a mechanism.
    Returns ``""`` when there is nothing to add (no report, or zero clusters)."""
    context = getattr(report, "as_hypothesis_context", lambda: "")()
    if not context:
        return ""
    return "\n" + context + "\n"


def _extract_referenced(raw: str, ctx: "ExploreContext | None") -> list[str]:
    """Best-effort, DISPLAY-ONLY list of explore artifacts M3's output referenced.

    Matches chart titles/names mentioned in the judge output as a whole phrase
    (word boundaries, min length) to avoid crediting a chart just because a short
    generic title like "rate"/"loss" happens to be a substring of unrelated prose.
    Used only for provenance/dashboard display — no effect on hypothesis survival.
    """
    if ctx is None or ctx.is_empty:
        return []
    import re

    low = str(raw).lower()
    referenced: list[str] = []
    for c in ctx.charts:
        for key in ("title", "name"):
            label = str(c.get(key) or "").strip()
            # Require a reasonably specific label matched on word boundaries.
            if len(label) >= 6 and label not in referenced:
                if re.search(rf"\b{re.escape(label.lower())}\b", low):
                    referenced.append(label)
                    break
    return referenced


@dataclass
class DiagnosisResult:
    """Output of :class:`DiagnosisAgent`.

    Attributes:
        model_name:           ``repr()`` of the analysed model.
        hypotheses:           Proposed :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`
                              objects for M4.
        findings_summary:     The findings dict forwarded to the judge.
        raw_judge_output:     Verbatim LLM response (useful for debugging).
        referenced_charts:    Explore chart titles M3 cited (provenance only).
        explore_context_used: Whether an :class:`ExploreContext` was supplied.
        failure_modes_used:   Whether a non-empty ``FailureModeReport`` was supplied.
    """

    model_name: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    findings_summary: dict[str, Any] = field(default_factory=dict)
    raw_judge_output: str = ""
    prompt: str = ""
    referenced_charts: list[str] = field(default_factory=list)
    explore_context_used: bool = False
    failure_modes_used: bool = False
    #: The adversarial critic's verbatim verdicts (``KEEP:``/``REJECT:`` +
    #: ``REASON:`` lines) and its tally. The critic ANNOTATES hypotheses
    #: (``metadata["critic"]``, ``metadata["critic_reason"]``) and orders the
    #: kept ones first — it never removes a proposal: with M5 testing on the
    #: held-out split, the data is the arbiter, and a critic that rejects every
    #: claim used to end the run with "0 hypotheses" and no M5/M4/fix at all.
    critic_raw_output: str = ""
    critic_prompt: str = ""
    n_critic_kept: int = 0
    n_critic_rejected: int = 0
    # M3 is two separate model calls: a proposer and an adversarial reviewer.
    # The proposer's full list and the reviewer's per-hypothesis decisions are
    # kept as well, so a rejected proposal stays distinguishable from a
    # parser/model failure in the report and in Langfuse. ``review_prompt`` /
    # ``review_raw`` mirror ``critic_prompt`` / ``critic_raw_output``;
    # ``review_decisions`` (statement, decision, reason) is derived from the
    # critic annotations — decision keep | reject | unparsed |
    # review_unavailable — and the reviewer never removes a hypothesis.
    proposed_hypotheses: list[Hypothesis] = field(default_factory=list)
    review_prompt: str = ""
    review_raw: str = ""
    review_decisions: list[dict[str, Any]] = field(default_factory=list)


_HYPOTHESIS_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["hypothesis", "failure_mode"],
        "properties": {
            "hypothesis":   {"type": "string", "minLength": 10},
            "plain_statement": {"type": "string"},
            "failure_mode": {"type": "string", "minLength": 2},
            "expected_association": {"type": "string"},
        },
    },
}


def _parse_hypotheses_json(raw: str, model_name: str) -> list[Hypothesis] | None:
    """Try to parse *raw* as a JSON array of hypothesis objects.

    Returns ``None`` when the response is not JSON or fails schema validation
    so the caller can fall back to the text parser.
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        import re as _re
        text = _re.sub(r"^```\w*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    errors = validate_json_shape(data, _HYPOTHESIS_SCHEMA)
    if errors:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "DiagnosisAgent JSON schema validation failed: %s", errors
        )
        return None

    return [
        Hypothesis(
            statement=item["hypothesis"],
            target_model=model_name,
            predicted_failure_mode=item["failure_mode"],
            plain_statement=str(item.get("plain_statement") or item.get("plain_language") or item.get("plain") or ""),
            test_design=str(item.get("test", "")),
            expected_association=str(item.get("expected_association", "")),
        )
        for item in data
        if item.get("hypothesis") and item.get("failure_mode")
    ]


# A judge that answers in Markdown decorates the labels — ``**HYPOTHESIS:**``,
# ``**HYPOTHESIS**:``, ``- HYPOTHESIS:``, ``1. TEST:``, ``### FAILURE_MODE:`` —
# and the bare ``startswith("HYPOTHESIS:")`` check saw none of them: a live
# qwen3.5-2b/bbh_word_sorting run had three well-formed hypotheses in the
# response and M3 reported zero. Normalise the label, leave the text alone.
#
# The ordinal AFTER the label is the same bug and was missed the first time.
# ``HYPOTHESIS 1:`` is how a model numbers a list of three of them — the most
# natural formatting there is, and the one Opus used on a 560-case audio-visual
# run: three hypotheses, each with a test design naming a BH-surviving signal,
# and M3 again reported zero. Two and a half hours of GPU and a high-effort
# judge, discarded over a digit. Accept the ordinal in either position.
_LABEL_LINE = re.compile(
    r"^\s*(?:(?:[-*•>]|#+|\d+[.)])\s*)*[*_`]*\s*"
    r"(HYPOTHESIS|PLAIN_STATEMENT|PLAIN_LANGUAGE|FAILURE_MODE|TEST|EXPECTED_ASSOCIATION|KEEP|REJECT|REASON)"
    r"(?:\s*[#(]?\d+[.)]?)?"        # HYPOTHESIS 1:  /  TEST #2:  /  REASON (3):
    r"\s*(?P<close>[*_`]*)\s*:\s*(?P<after>[*_`]*)\s*",
    re.IGNORECASE,
)


#: A leading emphasis/code marker, past any list bullet or heading. Its presence
#: is what makes a marker after the colon a CLOSER rather than content.
_OPENING_MARKER = re.compile(r"^\s*(?:(?:[-*•>]|#+|\d+[.)])\s*)*[*_`]")


#: A value that is ENTIRELY one code span or emphasis pair: `mode`, **mode**.
_WRAPPED_VALUE = re.compile(r"^([*_`]{1,2})(.+?)\1$")


def _unwrap_value(value: str) -> str:
    """Strip wrapping decoration from a value that is a bare identifier.

    Applied to FAILURE_MODE and EXPECTED_ASSOCIATION only. Those are looked up
    verbatim -- ``_FAILURE_MODE_TO_ANALYZERS["ignored_obs"]`` decides which
    analyzers the next cycle re-probes with -- so a judge writing
    ``FAILURE_MODE: `ignored_obs` `` must not produce a key with a backtick in
    it. Prose fields (HYPOTHESIS, TEST) keep their code spans: there the
    backticks mark identifiers inside a sentence and removing them was its own
    bug.

    Only an ENTIRELY wrapped value is unwrapped, so ``some `sig` text`` is left
    exactly as written.
    """
    text = value.strip()
    m = _WRAPPED_VALUE.match(text)
    return m.group(2).strip() if m else text


def _normalise_label_line(line: str) -> str:
    """``**HYPOTHESIS:** foo`` / ``- FAILURE_MODE: bar`` → ``HYPOTHESIS: foo`` /
    ``FAILURE_MODE: bar``. Lines without a recognised label are returned
    stripped and otherwise untouched (emphasis inside the statement stays)."""
    line = line.strip()
    m = _LABEL_LINE.match(line)
    if not m:
        return line

    # Decoration is only decoration when something opened it. `**KEEP:** x` and
    # `` `KEEP:` x `` wrap the LABEL, so the marker after the colon closes that
    # wrapper and goes. `TEST: `modality_ablation.grounded_in_audio` ...` opens a
    # code span belonging to the CONTENT, and eating it corrupts the identifier
    # M5 routes on. The difference is whether the line opened with a marker at
    # all -- so decide on that rather than on the marker's position.
    opened = bool(_OPENING_MARKER.match(line))
    rest = line[m.end():] if opened else (m.group("after") or "") + line[m.end():]
    rest = rest.strip()
    if opened:
        rest = re.sub(r"[*_`]+$", "", rest).strip()
    return f"{m.group(1).upper()}: {rest}"


def _parse_hypotheses(raw: str, model_name: str) -> list[Hypothesis]:
    """Parse hypothesis objects from LLM output.

    Tries JSON structured format first (more reliable), then falls back to the
    line-oriented ``HYPOTHESIS:`` / ``FAILURE_MODE:`` text format.  This makes
    the parser resilient to both output modes without requiring a hard migration.
    Labels may carry Markdown decoration (see :func:`_normalise_label_line`).
    """
    json_result = _parse_hypotheses_json(raw, model_name)
    if json_result is not None:
        return json_result

    # Text-format fallback
    hypotheses: list[Hypothesis] = []
    statement: str | None = None
    plain_statement: str = ""
    for line in raw.splitlines():
        line = _normalise_label_line(line)
        if line.upper().startswith("HYPOTHESIS:"):
            statement = line[len("HYPOTHESIS:"):].strip()
            plain_statement = ""
        elif (line.upper().startswith("PLAIN_STATEMENT:") or line.upper().startswith("PLAIN_LANGUAGE:")):
            tag = "PLAIN_STATEMENT:" if line.upper().startswith("PLAIN_STATEMENT:") else "PLAIN_LANGUAGE:"
            plain_statement = line[len(tag):].strip()
            if hypotheses and not hypotheses[-1].plain_statement:
                hypotheses[-1].plain_statement = plain_statement
        elif line.upper().startswith("FAILURE_MODE:") and statement:
            mode = _unwrap_value(line[len("FAILURE_MODE:"):])
            hypotheses.append(
                Hypothesis(
                    statement=statement,
                    target_model=model_name,
                    predicted_failure_mode=mode,
                    plain_statement=plain_statement,
                )
            )
            statement = None
            plain_statement = ""
        elif line.upper().startswith("TEST:") and hypotheses:
            # Attach the test design to the most recent hypothesis.
            hypotheses[-1].test_design = line[len("TEST:"):].strip()
        elif line.upper().startswith("EXPECTED_ASSOCIATION:") and hypotheses:
            hypotheses[-1].expected_association = _unwrap_value(
                line[len("EXPECTED_ASSOCIATION:"):]
            ).lower()
    return hypotheses


def _format_label_context(cases: "Any | None") -> str:
    """Descriptive label summary for the M3 critic (and nothing else).

    The critic used to see only the findings JSON — per-case rows with
    ``labelled_fail`` and ``extracted_answer`` but no gold — and rejected a
    correct "answers Yes regardless of the audio" hypothesis for "no
    ground-truth present/absent field in the evidence" (audiocaps 2026-08-20)
    while the proposer had read exactly that breakdown in the explore notes.
    Counts only; on a yes/no (true/false) batch a gold x answer table whose
    cells carry the FAIL count. Empty string when *cases* is ``None`` or
    carries no labels, so callers can concatenate it unconditionally.
    """
    if cases is None:
        return ""
    try:
        from evalvitals.analyzers.reasoning._text import (
            binary_direction,
            binary_gold,
            extract_answer,
        )
        from evalvitals.core.case import Label
    except Exception:  # pragma: no cover - defensive import guard
        return ""
    rows = [c for c in cases if getattr(c, "label", None) in (Label.PASS, Label.FAIL)]
    if not rows:
        return ""
    n_fail = sum(1 for c in rows if c.label == Label.FAIL)
    lines = [f"  labelled cases: {len(rows)} ({n_fail} FAIL / {len(rows) - n_fail} PASS)"]
    golds = [(c, binary_gold(getattr(c, "expected", None))) for c in rows]
    binary = [(c, g) for c, g in golds if g is not None]
    if len(binary) >= max(4, int(0.8 * len(rows))):
        table: dict[tuple[str, str], list[int]] = {}
        for c, g in binary:
            text = str(getattr(c, "observed", "") or "")
            a = binary_direction(extract_answer(text)) or binary_direction(text[-200:], last=True)
            key = (g, a or "unparsed")
            cell = table.setdefault(key, [0, 0])
            cell[0] += 1
            cell[1] += int(c.label == Label.FAIL)
        lines.append("  yes/no task -- gold x answer (n, of which FAIL):")
        for g in ("yes", "no"):
            for a in ("yes", "no", "unparsed"):
                n, f = table.get((g, a), [0, 0])
                if n:
                    lines.append(f"    gold={g:<3} answered={a:<8} n={n:<4} FAIL={f}")
        n_yes = sum(n for (g, a), (n, f) in table.items() if a == "yes")
        lines.append(f"  answered yes on {n_yes}/{len(binary)} binary cases")
    return "LABEL SUMMARY (descriptive counts from the case batch, not a test result):\n" + "\n".join(lines) + "\n"


def _validate_hypotheses(
    hypotheses: list[Hypothesis],
    findings_json: str,
    judge: "Model",
    *,
    capture: "dict[str, Any] | None" = None,
    context: str = "",
) -> list[Hypothesis]:
    """Adversarial review of *hypotheses*: a critic call at temperature=0.

    The same judge is called a second time with a prompt designed to find
    reasons to *reject* each hypothesis, breaking the confirmation-bias loop
    where the model that generated a claim then waves it through.

    The critic ANNOTATES, it does not filter: every hypothesis comes back,
    ``metadata["critic"]`` set to ``"keep"`` / ``"reject"`` (``"unparsed"``
    when the critic's answer named neither), ``metadata["critic_reason"]``
    carrying its ``REASON:`` line, kept hypotheses ordered first. Dropping
    the rejected ones used to end a run at "0 hypotheses" whenever a strict
    critic (opus at high effort, n=32 evidence) rejected all of them —
    three llm_benchmark runs in a row on 2026-08-20 — with no M5, M4 or fix
    afterwards. M5 now tests on the held-out split, so the data decides; the
    critic's verdict travels with the hypothesis as provenance and as the
    tie-breaker for M4's best-lead ordering.

    *capture*, when given, receives ``raw`` (the critic's verbatim answer),
    ``prompt``, ``n_kept`` and ``n_rejected``. Falls back to the unannotated
    list if the critic call fails so the loop is never blocked by a transient
    error.

    *context* is the evidence the PROPOSER worked from beyond the findings
    JSON -- M2's conclusion and evidence chain, the statistical verdicts, the
    explore notes and the label summary. Without it the critic judged the
    hypotheses against less than the proposer had seen and rejected them for
    missing evidence that was on the table (audiocaps 2026-08-20).
    """
    if not hypotheses:
        return hypotheses

    hyp_lines = "\n".join(
        f"- HYPOTHESIS: {h.statement}  (failure_mode: {h.predicted_failure_mode})"
        for h in hypotheses
    )
    context_section = (
        "\nContext the proposer worked from (analyst conclusion, evidence chain, "
        "statistical verdicts, exploratory notes, label summary):\n"
        + context.strip() + "\n"
    ) if context and context.strip() else ""
    prompt = _VALIDATE_PROMPT.format(
        context_section=context_section,
        findings_json=findings_json,
        hypotheses_text=hyp_lines,
    )
    if capture is not None:
        capture["prompt"] = prompt

    try:
        # Use temperature=0 so the critic is deterministic and strict.
        # Models that don't accept a temperature kwarg are called without it.
        import inspect
        sig = inspect.signature(judge.generate)
        if "temperature" in sig.parameters:
            raw = judge.generate(prompt, temperature=0)
        else:
            raw = judge.generate(prompt)
    except Exception:
        if capture is not None:
            # A transport failure must not silently delete ideas — and the
            # record must distinguish it from an evidence-based rejection.
            capture["decisions"] = [{
                "statement": h.statement, "decision": "review_unavailable",
                "reason": "The adversarial review call failed; proposal retained.",
            } for h in hypotheses]
        return hypotheses  # validation failed — keep originals, unannotated
    raw = str(raw)
    if capture is not None:
        capture["raw"] = raw

    def _match(stmt: str) -> "Hypothesis | None":
        stmt = stmt.strip().lower()
        for h in hypotheses:
            hs = h.statement.lower()
            if hs[:60] in stmt or stmt[:60] in hs:
                return h
        return None

    verdicts: dict[int, str] = {}
    reasons: dict[int, str] = {}
    last: "Hypothesis | None" = None
    saw_decision = False
    for line in raw.splitlines():
        line = _normalise_label_line(line)
        upper = line.upper()
        if upper.startswith("KEEP:") or upper.startswith("REJECT:"):
            saw_decision = True
            verdict = "keep" if upper.startswith("KEEP:") else "reject"
            h = _match(line.split(":", 1)[1])
            if h is not None:
                verdicts[id(h)] = verdict
                last = h
            else:
                last = None
        elif upper.startswith("REASON:") and last is not None:
            reasons[id(last)] = line[len("REASON:"):].strip()

    if not saw_decision:
        # Malformed/empty critic output is an infrastructure failure: keep the
        # proposals, say so, and mark them so the record shows no review ran.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "DiagnosisAgent validation: critic response could not be parsed for "
            "%d hypothesis(es) — keeping them unreviewed",
            len(hypotheses),
        )
        for h in hypotheses:
            h.metadata["critic"] = "unparsed"
        if capture is not None:
            capture.update(n_kept=0, n_rejected=0, decisions=_critic_decisions(hypotheses))
        return hypotheses

    for h in hypotheses:
        h.metadata["critic"] = verdicts.get(id(h), "unparsed")
        if id(h) in reasons:
            h.metadata["critic_reason"] = reasons[id(h)]
    n_kept = sum(1 for h in hypotheses if h.metadata.get("critic") == "keep")
    n_rejected = sum(1 for h in hypotheses if h.metadata.get("critic") == "reject")
    if capture is not None:
        capture.update(n_kept=n_kept, n_rejected=n_rejected,
                       decisions=_critic_decisions(hypotheses))
    if n_rejected and not n_kept:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "DiagnosisAgent validation: critic rejected all %d hypothesis(es) — "
            "kept as flagged leads; M5 testing decides",
            len(hypotheses),
        )
    order = {"keep": 0, "unparsed": 1, "reject": 2}
    return sorted(hypotheses, key=lambda h: order.get(h.metadata.get("critic"), 1))


def _critic_decisions(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
    """Per-hypothesis review record derived from the critic's annotations: the
    ``{statement, decision, reason}`` shape the run log, report and Langfuse
    consume. ``decision`` is ``keep`` / ``reject`` / ``unparsed`` (or
    ``review_unavailable`` when the critic call itself failed). The critic
    never removes a hypothesis, so this is a record, not a filter."""
    return [{
        "statement": h.statement,
        "decision": str(h.metadata.get("critic", "unparsed")),
        "reason": str(h.metadata.get("critic_reason", "")),
    } for h in hypotheses]


def _default_judge() -> "Model":
    """Return a judge model without requiring an explicit API key.

    Resolution order:
    1. ``agy`` CLI (antigravity) — if the binary is on PATH, no key needed.
    2. Gemini — if ``GEMINI_API_KEY`` is set.
    3. Raise with instructions.
    """
    import os
    import shutil

    if shutil.which("agy"):
        from evalvitals.agent_runtime.judges import AgyModel
        return AgyModel()

    if os.getenv("GEMINI_API_KEY"):
        from evalvitals.models.blackbox.gemini import GeminiModel
        return GeminiModel()

    raise ValueError(
        "DiagnosisAgent requires a judge model. "
        "Options: install antigravity (agy) — no API key needed, "
        "or set GEMINI_API_KEY to use Gemini "
        "(install with: pip install 'evalvitals[gemini]'), "
        "or pass judge= explicitly."
    )


class DiagnosisAgent:
    """Proposes hypotheses from an :class:`~evalvitals.eval_agent.analysis.AnalysisReport`.

    Args:
        judge:    Any :class:`~evalvitals.core.model.Model` with
                  ``Capability.GENERATE``.  Defaults to a ``GeminiModel``
                  when ``GEMINI_API_KEY`` is in the environment.
        api_key:  Gemini API key — used only when *judge* is ``None`` and no
                  ``GEMINI_API_KEY`` env var is set.
        model_id: Gemini model identifier (default: ``"gemini-2.0-flash"``).
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        api_key: str | None = None,
        model_id: str = "gemini-2.0-flash",
    ) -> None:
        if judge is None and api_key is not None:
            from evalvitals.models.blackbox.gemini import GeminiModel

            judge = GeminiModel(model_id=model_id, api_key=api_key)
        self._judge = judge  # None → resolved lazily on first call

    @property
    def judge(self) -> "Model":
        if self._judge is None:
            self._judge = _default_judge()
        return self._judge

    @judge.setter
    def judge(self, value: "Model") -> None:
        self._judge = value

    def _repair_plain_language(
        self, raw: str, hypotheses: list[Hypothesis], model_name: str,
    ) -> list[Hypothesis]:
        """One bounded retry when a PLAIN_STATEMENT line is missing or jargon-y.

        The prompt asking for plain language is not enough on its own — a judge
        will happily reuse the technical line — so the same host-side check the
        explorer and the analysis-side hypothesis agent already run
        (:func:`~evalvitals.analysis.plain_language.jargon_violation`) also
        guards this path, which is the one the benchmark runner takes. Never
        raises: a failed repair keeps the original hypotheses, since a jargon-y
        headline is worse than no diagnosis only for the reader, not for M5.
        """
        if not hypotheses:
            return hypotheses
        violations = [
            f"hypothesis {i}'s PLAIN_STATEMENT {reason}: {h.plain_statement!r}"
            for i, h in enumerate(hypotheses, start=1)
            for reason in [jargon_violation(h.plain_statement, h.statement)]
            if reason
        ]
        if not violations:
            return hypotheses
        try:
            repaired_raw = self.judge.generate(_PLAIN_REPAIR_PROMPT.format(
                raw=raw, violations="\n".join(f"- {v}" for v in violations),
            ))
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "DiagnosisAgent: plain-language repair failed: %s", exc)
            return hypotheses
        repaired = _parse_hypotheses(str(repaired_raw), model_name)
        return repaired or hypotheses

    def diagnose(
        self,
        analysis: "AnalysisReport | dict[str, Result]",
        model_name: str = "",
        prior_cycles: list[dict] | None = None,
        explore_context: "ExploreContext | None" = None,
        failure_modes: "Any | None" = None,
        cases: "Any | None" = None,
    ) -> DiagnosisResult:
        """Synthesize *analysis* into a set of falsifiable hypotheses.

        Args:
            analysis:     An :class:`~evalvitals.eval_agent.analysis.AnalysisReport`
                          from M2, or (for backward compatibility) a plain
                          ``{analyzer_name: Result}`` dict.
            model_name:   Ignored when *analysis* is an ``AnalysisReport``
                          (the name is taken from the report).
            prior_cycles: Summary of previous M1→M4 cycles produced by
                          :class:`~evalvitals.eval_agent.legacy.AutoDiagnoseLoop`.
                          Each entry is ``{"cycle": int, "severity": str,
                          "hypotheses": [{"statement", "failure_mode", "status"}]}``.
                          Injected into the prompt so the judge avoids re-proposing
                          already-tested hypotheses.
            failure_modes: Optional :class:`~evalvitals.analysis.failure_modes.FailureModeReport`
                          (clustered FAIL cases). Descriptive only, like
                          *explore_context* — informs which hypotheses M3
                          proposes, never a claim itself. ``None`` (default)
                          adds nothing to the prompt and costs no extra call.
            cases:        The labelled case batch M1/M2 ran on. Used ONLY to
                          give the critic a label summary (PASS/FAIL counts;
                          gold x answer table on a yes/no batch) — see
                          :func:`_format_label_context`. ``None`` adds nothing.

        Returns:
            :class:`DiagnosisResult` with zero or more hypotheses.
        """
        from evalvitals.analysis.analysis_module import AnalysisModule, AnalysisReport

        if not isinstance(analysis, AnalysisReport):
            # Backward compat: wrap raw results in a minimal AnalysisReport.
            analysis = AnalysisModule().analyze(analysis, model_name)

        summary = {name: r.findings for name, r in analysis.raw_results.items()}

        # Prefer M2's LLM conclusion + evidence chain (StatsAnalysisReport) over
        # the bare threshold narrative, so a real failure mode surfaced by the
        # analyst is not dropped just because no numeric threshold fired.
        conclusion = getattr(analysis, "conclusion", "") or analysis.narrative or "(no conclusion)"
        evidence_chain = getattr(analysis, "evidence_chain", None) or []
        evidence_section = ""
        if evidence_chain:
            evidence_section = (
                "\nEvidence chain:\n"
                + "\n".join(f"  - {step}" for step in evidence_chain)
                + "\n"
            )
        stats_results = getattr(analysis, "stats_results", None) or []
        stats_section = ""
        stats_lines = [
            f"  - {r.summary}"
            for r in stats_results
            if getattr(r, "ok", False) and getattr(r, "summary", "")
        ]
        if stats_lines:
            stats_section = "\nStatistical test results:\n" + "\n".join(stats_lines) + "\n"

        # Evidence sources M3 can reference in TEST designs: per-case signal
        # keys and strategy contrasts that already exist in this cycle.
        signals: list[str] = []
        for r in stats_results:
            cfg = getattr(r, "config", None) or {}
            sig = cfg.get("signal")
            if sig and sig not in signals:
                signals.append(sig)
            for s in cfg.get("strategies", []) or []:
                tag = f"strategy contrast vs '{s}'"
                if s != "baseline" and tag not in signals:
                    signals.append(tag)
        available_signals_section = ""
        if signals:
            available_signals_section = (
                "AVAILABLE EVIDENCE (per-case signals / contrasts already measured):\n"
                + "\n".join(f"  - {s}" for s in signals)
                + "\n\n"
            )

        prompt = _DIAGNOSE_PROMPT.format(
            prior_section=_format_prior_section(prior_cycles or []),
            model_name=analysis.model_name or model_name,
            severity=analysis.severity,
            conclusion=conclusion,
            evidence_section=evidence_section,
            stats_section=stats_section,
            explore_section=_format_explore_section(explore_context),
            failure_modes_section=_format_failure_modes_section(failure_modes),
            available_signals_section=available_signals_section,
            findings_json=json.dumps(summary, indent=2, default=str),
        )
        import inspect as _inspect
        from pathlib import Path as _Path
        # M2's confirmatory figures, then the explorer's (UNCONFIRMED) charts.
        # The explore charts are attached to the M3 prompt ONLY — they never reach
        # M2's confirmatory family, M5, or the fix gate.
        _figs = [_Path(f) for f in getattr(analysis, "figures", []) if _Path(f).exists()]
        if explore_context is not None:
            _figs += [_Path(f) for f in explore_context.figure_paths if _Path(f).exists()]
        _sig = _inspect.signature(self.judge.generate)
        if "images" in _sig.parameters and _figs:
            raw = self.judge.generate(prompt, images=_figs)
        else:
            raw = self.judge.generate(prompt)
        proposed_hypotheses = _parse_hypotheses(str(raw), analysis.model_name or model_name)
        proposed_hypotheses = self._repair_plain_language(
            str(raw), proposed_hypotheses, analysis.model_name or model_name,
        )
        hypotheses = list(proposed_hypotheses)

        # Adversarial validation: run a second critic call at temperature=0 to
        # prune hypotheses the generator produced without sufficient evidence.
        # This prevents the self-evaluation loop where the same model that
        # proposed a hypothesis then approves it uncritically.
        critic: dict[str, Any] = {}
        if hypotheses:
            findings_json_str = json.dumps(summary, indent=2, default=str)
            # The critic reviews against what the proposer saw, not less:
            # conclusion + evidence chain + stats verdicts + explore notes,
            # plus a label summary the proposer never had either.
            critic_context = "\n".join(part.strip("\n") for part in (
                f"Analysis conclusion: {conclusion}",
                evidence_section,
                stats_section,
                _format_explore_section(explore_context),
                _format_label_context(cases),
            ) if part and part.strip())
            hypotheses = _validate_hypotheses(
                hypotheses, findings_json_str, self.judge, capture=critic,
                context=critic_context,
            )

        # Fallback: if the judge returned NO_ISSUE but M2 has medium/high findings,
        # auto-generate one hypothesis per finding so M4 can still run.
        # This prevents self-diagnosis bias when the judge is the model under test.
        if not hypotheses and analysis.findings:
            from evalvitals.analysis.analysis_module import _SEVERITY_ORDER
            raw_text = str(raw or "").strip()
            if raw_text and "NO_ISSUE" not in raw_text.upper():
                # Not a NO_ISSUE verdict: the judge proposed something the label
                # parser could not read (gemma-4-e2b/bbh_causal_judgement 2026-08-22:
                # ``**HYPOTHESIS 1:**``). Substituting templates silently turned a
                # three-hypothesis diagnosis into "low self-consistency".
                import logging

                logging.getLogger(__name__).warning(
                    "DiagnosisAgent: the judge wrote %d chars that parsed to zero hypotheses "
                    "(not NO_ISSUE); falling back to analysis-module template hypotheses — "
                    "check the label format in the response: %r",
                    len(raw_text), raw_text[:160],
                )
            for finding in analysis.findings:
                if _SEVERITY_ORDER.get(finding.severity, 0) >= 2:  # medium or high
                    hypotheses.append(Hypothesis(
                        statement=(
                            f"The model {finding.message} "
                            f"({finding.analyzer}.{finding.metric}={finding.value:.3g})"
                        ),
                        target_model=analysis.model_name or model_name,
                        predicted_failure_mode=finding.analyzer,
                    ))

        return DiagnosisResult(
            model_name=analysis.model_name or model_name,
            hypotheses=hypotheses,
            findings_summary=summary,
            raw_judge_output=str(raw),
            prompt=prompt,
            referenced_charts=_extract_referenced(str(raw), explore_context),
            explore_context_used=bool(explore_context is not None and not explore_context.is_empty),
            failure_modes_used=bool(getattr(failure_modes, "clusters", None)),
            critic_raw_output=str(critic.get("raw", "") or ""),
            critic_prompt=str(critic.get("prompt", "") or ""),
            n_critic_kept=int(critic.get("n_kept", 0) or 0),
            n_critic_rejected=int(critic.get("n_rejected", 0) or 0),
            # The same review in the proposer/reviewer record shape: the full
            # proposal list and a per-hypothesis decision derived from the
            # critic annotations (the critic never removes a hypothesis).
            proposed_hypotheses=proposed_hypotheses,
            review_prompt=str(critic.get("prompt", "") or ""),
            review_raw=str(critic.get("raw", "") or ""),
            review_decisions=list(critic.get("decisions") or []),
        )
