"""RunLogger — structured per-cycle logging for AutoDiagnoseLoop.

Writes a JSONL event log (one line per M1/M2/M3/M5 event) and saves heavy
analyzer artifacts (attention tensors, hidden-state arrays) to a separate
``artifacts/`` directory, keyed by cycle number so they stay navigable.

Each JSON record always contains a ``ts`` (ISO-8601), an ``event`` field, and a
``schema_version`` (int) — bumped whenever an event's fields are renamed,
removed, or change meaning, so a parser can detect breaking changes without
guessing from ``evalrx_version``. See ``RUN_LOG_SCHEMA_VERSION`` below.
The underlying file handler is a standard :class:`logging.FileHandler`, so
callers can attach additional handlers (e.g. a ``StreamHandler`` for console
output) by accessing :attr:`RunLogger.logger`.

Usage::

    from evalrx.eval_agent import AutoDiagnoseLoop, RunLogger

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger("runs/exp_01"))
    report = loop.run(cases)
    # runs/exp_01/run_log.jsonl          ← one JSON line per event, grep/jq friendly
    # runs/exp_01/artifacts/c0_attention_attn_weights.npy  ← attention tensor
    # runs/exp_01/artifacts/c0_cka_layer_similarities.npy  ← CKA matrix

Stream events while running::

    tail -f runs/exp_01/run_log.jsonl | python -m json.tool

Filter by module::

    jq 'select(.event=="diagnosis")' runs/exp_01/run_log.jsonl

Auto-timestamped run dir (default when no path is given)::

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger())
    # loop.run_logger.run_dir  → runs/20260603_142305/

Verbose console output (human-readable summary to stdout)::

    loop = AutoDiagnoseLoop(model=model, run_logger=RunLogger(verbose=True))

Custom handler — e.g. redirect verbose output to a file instead::

    import logging, sys
    rl = RunLogger("runs/exp_01")
    rl.logger.addHandler(logging.StreamHandler(sys.stderr))
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import sys
import textwrap
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.analysis.analysis_module import AnalysisReport
    from evalrx.core.result import Result
    from evalrx.eval_agent.hypothesis import Hypothesis
    from evalrx.eval_agent.loop_reports import AutoDiagnoseReport
    from evalrx.eval_agent.stages.diagnosis import DiagnosisResult
    from evalrx.eval_agent.stages.surgery import InterventionResult

# Bump when an existing event's fields are renamed, removed, or change meaning,
# or when a new event TYPE is added (additive fields on an existing event don't
# need a bump). Downstream parsers of run_log.jsonl can branch on this instead
# of guessing from `evalrx_version`.
# v2: `analysis`'s stats_tool_results/stats_results/stats_plan/
#     corrected_rejections are now conditionally externalized (see
#     _externalize_if_large) — a {path, n_items, bytes} summary instead of
#     the raw value once it exceeds _INLINE_MAX_BYTES.
# v3: two new event types for AgenticDiagnoseLoop — `agent_decision` (one judge
#     decision turn: chosen tool + rationale, keyed by `step` not `cycle`) and
#     `agent_tool` (the dispatch layer's accept/reject outcome for that tool
#     call). VLDiagnoseLoop/AutoDiagnoseLoop's events are unchanged.
# v5: the EvalVitals → EvalRX rename renamed the `evalvitals_version` field to
#     `evalrx_version` on every event (see build_envelope below) and moved the
#     published schema's `$id` to https://evalrx.dev/schemas/run_log.schema.json.
#     Old run_log.jsonl files still parse (the schema is permissive on unknown
#     fields) but a strict `evalrx_version`-keyed reader needs this bump to
#     tell them apart from pre-rename logs.
RUN_LOG_SCHEMA_VERSION = 5


def _externalized(summary: "dict[str, Any]") -> str:
    """Render an ``_externalize_if_large`` stand-in instead of iterating it.

    Says where the payload went rather than dropping the line: an externalised
    value is exactly the case where the data is most worth pointing at.
    """
    where = summary.get("path") or "artifacts/"
    return f"({summary.get('n_items', '?')} items externalised -> {where})"


#: Plain-language glosses for the console narration.
#:
#: The console is read while a run is in flight, by the same person the report
#: is written for: someone who builds evaluations and does not do statistics.
#: `status=supported effect=0.283` is an audit record, not a sentence, so every
#: stage line that carries a verdict gets one plain lead ABOVE the raw fields —
#: the raw fields stay, because they are what a bug report needs.

def _odds_phrase(e_value: Any) -> str:
    """An e-value as betting odds, or "" when there is no usable number.

    An e-value IS odds against the null: e=45 means the evidence runs about 45
    to 1 against this being chance. "Reject at alpha=0.05" is the same statement
    in a dialect nobody outside the field speaks. Mirrors the wording the report
    UI uses for the same number so the two never disagree.
    """
    try:
        e = float(e_value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(e) or e <= 0:
        return ""
    if e >= 1000:
        return "over 1000 to 1"
    return f"about {round(e)} to 1" if e >= 10 else f"about {e:.1f} to 1"


def _flip_phrase(payload: "dict[str, Any]") -> str:
    """"fixed 45 and broke 6 of the 125 cases it was tested on", or ""."""
    if payload.get("n_fixed") is None and payload.get("n_broken") is None:
        return ""
    fixed = int(payload.get("n_fixed") or 0)
    broken = int(payload.get("n_broken") or 0)
    pairs = int(payload.get("n_pairs") or 0)
    tail = f" of the {pairs} case{'' if pairs == 1 else 's'} it was tested on" if pairs else ""
    return f"fixed {fixed} and broke {broken}{tail}"


def _case_snapshot(case: Any) -> "dict[str, Any]":
    """Make a small, renderer-safe baseline record for an evidence example."""
    if hasattr(case, "to_dict"):
        value = case.to_dict()
    elif isinstance(case, dict):
        value = dict(case)
    else:
        value = {"id": str(getattr(case, "id", ""))}
    inputs = value.get("inputs") if isinstance(value.get("inputs"), dict) else {}
    return {
        "id": str(value.get("id") or value.get("case_id") or ""),
        "input": inputs.get("prompt") or value.get("prompt") or value.get("instruction") or "",
        "baseline_output": value.get("observed", value.get("output")),
        "expected": value.get("expected"),
        "outcome": value.get("label") or value.get("status") or "unknown",
    }


def _iter_cases(cases: Any) -> "list[Any]":
    """Accept CaseBatch, a plain sequence, or a generator without assumptions."""
    if cases is None:
        return []
    value = getattr(cases, "cases", cases)
    try:
        return list(value)
    except TypeError:
        return []


def _probe_examples(results: "dict[str, Any]", cases: Any) -> "list[dict[str, Any]]":
    """Persist two real, bounded M1 walkthroughs beside aggregate findings.

    A probe only becomes a before/after comparison when its analyzer explicitly
    records both outputs.  Otherwise this records an honest *baseline case +
    check result* example; downstream UI must not call it an intervention.
    """
    snapshots: dict[str, dict[str, Any]] = {}
    for case in _iter_cases(cases):
        snapshot = _case_snapshot(case)
        if snapshot["id"]:
            snapshots[snapshot["id"]] = snapshot
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, result in results.items():
        findings = getattr(result, "findings", {}) or {}
        rows = findings.get("per_case") or []
        if not isinstance(rows, list):
            continue
        rows = sorted(
            (row for row in rows if isinstance(row, dict)),
            key=lambda row: 0 if str(snapshots.get(str(row.get("sample_id") or row.get("case_id") or ""), {}).get("outcome", "")).lower() == "fail" else 1,
        )
        for row in rows:
            case_id = str(row.get("sample_id") or row.get("case_id") or "")
            snapshot = snapshots.get(case_id)
            if not snapshot or case_id in used:
                continue
            checked = {
                str(key).replace("_", " "): value for key, value in row.items()
                if key not in {"sample_id", "case_id"} and isinstance(value, (str, int, float, bool))
            }
            if not checked:
                continue
            used.add(case_id)
            output.append({
                "id": f"m1-{name}-{case_id}", "kind": "case_measurement", "case_id": case_id,
                "probe_title": str(name).replace("_", " ").title(),
                **snapshot, "check_result": checked,
                "plain_reading": "This one case illustrates the recorded check. The aggregate M1 result uses all measured cases.",
                "evidence_scope": "one recorded case within M1",
            })
            break
        if len(output) >= 2:
            break
    return output


def _artifact_to_numpy(artifact: Any) -> "Any | None":
    """Convert *artifact* to a numpy array, or return None if not possible.

    Handles: torch.Tensor, list[torch.Tensor] (e.g. per-layer attentions),
    and numpy arrays.  A list of tensors is stacked along a new first axis so
    that ``attentions`` (list of ``(heads, seq, seq)``) becomes
    ``(layers, heads, seq, seq)`` — a single array that retains all the data.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    if hasattr(artifact, "detach"):  # torch.Tensor
        return artifact.detach().cpu().float().numpy()
    if isinstance(artifact, np.ndarray):
        return artifact
    if isinstance(artifact, list) and artifact and hasattr(artifact[0], "detach"):
        try:
            import torch
            return torch.stack(artifact).detach().cpu().float().numpy()
        except Exception:  # noqa: BLE001
            return None
    return None


def _save_artifact_figure(artifact_dir: Path, stem: str, arr: Any) -> None:
    """Save a matplotlib figure of *arr* when the shape and stem are recognised.

    Dispatch table (first match wins):
    - 4-D + ``attn`` in stem → mean over (layers, heads) → 2-D heatmap
    - 3-D + ``attn`` in stem → mean over heads → 2-D heatmap
    - 2-D + heatmap keyword  → direct heatmap (viridis)
    - 1-D + curve keyword    → line plot
    Skips silently when matplotlib is unavailable or the shape is unrecognised.
    """
    try:
        import matplotlib.pyplot as plt
        plt.ioff()
    except ImportError:
        return

    key = stem.lower()
    # Skip logit arrays — (seq, vocab) shape is too large for a useful figure
    if "logit" in key:
        return

    _is_attn = any(k in key for k in ("attn", "attention"))

    fig = None
    try:
        ndim = arr.ndim
        if ndim == 4 and _is_attn:
            mat = arr.mean(axis=(0, 1))  # (layers, heads, seq, seq) → (seq, seq)
            n_layers, n_heads = arr.shape[0], arr.shape[1]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_layers}L × {n_heads}H)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 3 and _is_attn:
            mat = arr.mean(axis=0)  # (heads, seq, seq) → (seq, seq)
            n_heads = arr.shape[0]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_heads} heads)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and "diff" in key:
            # Signed difference map (e.g. FAIL-mean minus PASS-mean attention):
            # diverging colormap with symmetric limits so the sign is readable.
            bound = float(max(abs(arr.min()), abs(arr.max()))) or 1.0
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-bound, vmax=bound)
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and (_is_attn or any(k in key for k in ("rollout", "spatial", "map"))):
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="viridis", aspect="auto")
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 1 and any(k in key for k in ("entropy", "score", "prob", "weight", "rollout")):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(arr)
            ax.set_xlabel("position")
            ax.set_ylabel(stem)
            ax.set_title(stem)
            plt.tight_layout()

        if fig is not None:
            fig.savefig(artifact_dir / f"{stem}.png", dpi=100, bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    finally:
        if fig is not None:
            plt.close(fig)


class _JsonFormatter(logging.Formatter):
    """Format each LogRecord as a single JSON line using the ``_payload`` extra."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(getattr(record, "_payload", {}), default=str)


class _VerboseFormatter(logging.Formatter):
    """Format each LogRecord as a human-readable stage summary.

    Reads the same ``_payload`` dict that ``_JsonFormatter`` serialises to
    JSON, dispatches on ``payload["event"]``, and returns a multi-line string
    with the most useful fields for interactive / Docker console output.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: PLR0911
        p = getattr(record, "_payload", {})
        event = p.get("event", "")
        cycle = p.get("cycle", "?")

        if event == "run_start":
            lines = ["\n[START] run configuration"]
            for k in (
                "model", "judge", "coder", "explorer", "max_cycles", "depth",
                "allow_codegen", "n_cases", "evalrx_version", "git_commit",
            ):
                if p.get(k) is not None:
                    lines.append(f"     {k:18s}: {p[k]}")
            return "\n".join(lines)

        if event == "probe":
            lines = [f"\n[M1] cycle={cycle}  analyzers={p.get('analyzers', [])}"]
            rationale = p.get("selection_rationale", "")
            if rationale:
                lines.append(f"     rationale  : {rationale}")
            for name, findings in (p.get("findings") or {}).items():
                scalars = {
                    k: round(v, 4)
                    for k, v in (findings or {}).items()
                    if isinstance(v, (int, float))
                }
                lines.append(f"     {name}: {dict(list(scalars.items())[:6])}")
            return "\n".join(lines)

        if event == "explore":
            status = "ok" if p.get("ok") else f"FAILED ({p.get('error', '?')})"
            lines = [
                f"\n[EXPLORE] cycle={cycle}  {status}  "
                f"observations={p.get('n_observations', 0)} "
                f"charts={p.get('n_charts_rendered', 0)}/{p.get('n_charts', 0)} rendered "
                f"tables={p.get('n_tables', 0)} "
                f"candidates={p.get('n_candidate_signals', 0)}"
            ]
            for obs in (p.get("observations") or [])[:4]:
                lines.append("     - " + textwrap.fill(str(obs), 72, subsequent_indent="       "))
            if p.get("out_dir"):
                lines.append(f"     out_dir    : {p['out_dir']}")
            lines.append("     (descriptive only — feeds M3's notes and the dashboard, "
                         "never M2/M4/fix)")
            return "\n".join(lines)

        if event == "analysis":
            lines = [f"\n[M2] cycle={cycle}  severity={p.get('severity')}"]
            if p.get("llm_fallback_reason"):
                lines.append(
                    f"     JUDGE FAILED — narrative below is the threshold "
                    f"fallback, not analysis: {p['llm_fallback_reason']}"
                )
            conclusion = p.get("conclusion")
            if conclusion:
                lines.append(
                    "     conclusion : "
                    + textwrap.fill(conclusion, 72, subsequent_indent="     ")
                )
            for step in (p.get("evidence_chain") or [])[:3]:
                lines.append(f"     evidence   : {step}")
            # These two are run through _externalize_if_large, which swaps an
            # oversized list for a {path, n_items, bytes} SUMMARY DICT. Iterating
            # that yields its keys — strings — so `s['tool']` raised TypeError
            # inside logging.emit, where Python swallows the exception: the run
            # carried on and the whole [M2] line vanished. Seen live on
            # qwen3.5-2b/minervamath, where M2's plan crossed the threshold.
            stats_plan = p.get("stats_plan") or []
            if isinstance(stats_plan, dict):
                lines.append(f"     stats_tools: {_externalized(stats_plan)}")
            elif stats_plan:
                lines.append(f"     stats_tools: {[s.get('tool') for s in stats_plan]}")
            corrected = p.get("corrected_rejections") or {}
            if isinstance(corrected, dict):
                # Per result, not per tool: every signal_label_assoc test shares
                # one tool name, so the tool list read "survived" for all of them.
                survivors = corrected.get("rejected_result_keys") or corrected.get("rejected_tools")
                if survivors:
                    n_tested = corrected.get("n_tested")
                    tested = f" of {n_tested}" if n_tested else ""
                    lines.append(f"     fdr_survive: {len(survivors)}{tested}: {survivors}")
                    # "3 of 17 survived e-BH" reads as a failure to anyone who
                    # has not met the correction. It is the opposite: the filter
                    # exists because testing 17 patterns at once turns up a few
                    # by luck alone, and these are the ones that outlived it.
                    lines.append(
                        f"     in plain terms: {len(survivors)}{tested} screened patterns "
                        "are still standing after discounting for how many were "
                        "tried at once; the rest could be luck"
                    )
            tool_results = p.get("stats_tool_results") or []
            if isinstance(tool_results, dict):
                lines.append(f"     stats_tool : {_externalized(tool_results)}")
            else:
                for tool in tool_results[:2]:
                    lines.append(
                        f"     stats_tool : {tool.get('name')} - {tool.get('conclusion', '')}"
                    )
            for fig in p.get("figures") or []:
                lines.append(f"     figure     : {fig}")
            if not conclusion:
                lines.append(
                    "     "
                    + textwrap.fill(p.get("narrative", ""), 72, subsequent_indent="     ")
                )
            return "\n".join(lines)

        if event == "diagnosis":
            head = f"\n[M3] cycle={cycle}  {p.get('n_hypotheses', 0)} hypothesis/es"
            n_rej = int(p.get("n_critic_rejected", 0) or 0)
            n_keep = int(p.get("n_critic_kept", 0) or 0)
            if n_rej or n_keep:
                head += f"  (critic: {n_keep} kept, {n_rej} rejected"
                head += (" — all rejected: kept as flagged leads, the held-out "
                         "M4 decides)" if n_rej and not n_keep else
                         "; rejected ones demoted)" if n_rej else ")")
            lines = [head]
            for h in p.get("hypotheses") or []:
                lines.append(f"     hypothesis  : {h.get('statement', '')}")
                lines.append(f"     failure_mode: {h.get('failure_mode', '')}")
                if h.get("critic"):
                    reason = (h.get("critic_reason") or "").strip()
                    lines.append(
                        f"     critic      : {h['critic']}"
                        + (f" — {reason[:160]}" if reason else "")
                    )
            return "\n".join(lines)

        if event == "surgery":
            module = p.get("module", "m5").upper()
            hyp = p.get("hypothesis", "")[:70]
            status = p.get("status", "?")
            lines = [f"\n[{module}] cycle={cycle}  '{hyp}'"]
            ev = p.get("evidence") or {}
            if module == "M4":
                # This is the only stage allowed to say whether an explanation
                # held up, so it is the line most worth being a sentence.
                plain = {
                    "supported": "the held-out half agreed with this explanation",
                    "refuted": "the held-out half pointed the other way",
                    "inconclusive": "the held-out half could not settle it either way",
                }.get(str(status).lower())
                if plain:
                    lines.append(f"     in plain terms: {plain}")
                lines.append(
                    f"     status={status}"
                    f"  effect={ev.get('m4_effect_size', '?')}"
                    f"  confidence={ev.get('m4_confidence', '?')}"
                )
                lines.append(
                    f"     protocol_consistent={ev.get('m4_protocol_consistent', '?')}"
                )
                lines.append(f"     verdict : {ev.get('m4_verdict', '')}")
            else:
                lines.append(f"     status={status}  fixed={p.get('fixed')}")
                if ev:
                    lines.append(f"     evidence: {dict(list(ev.items())[:4])}")
            return "\n".join(lines)

        if event == "experiment":
            module = p.get("module", "m5").upper()
            lines = [f"\n[{module}] cycle={cycle}  experiment run"]
            lines.append(f"     hypothesis : {p.get('hypothesis', '')[:70]}")
            lines.append(
                f"     status={p.get('status')}  verdict={p.get('verdict')}"
                f"  fixed={p.get('fixed')}  rc={p.get('returncode')}"
                f"  provider={p.get('provider')}"
            )
            code_paths = p.get("code_paths") or {}
            if code_paths:
                lines.append(f"     code       : {list(code_paths.values())}")
            ws = p.get("workspace_snapshot") or {}
            if ws.get("dir"):
                lines.append(
                    f"     workspace  : {ws['dir']} ({len(ws.get('files', []))} files)"
                )
            return "\n".join(lines)

        if event == "tool_codegen":
            ok = "OK" if p.get("ok") else "FAILED"
            lines = [
                f"\n[TOOL] cycle={cycle}  {p.get('module')}/{p.get('tool_name')}  "
                f"{ok}  source={p.get('source')}"
            ]
            if p.get("need"):
                lines.append(f"     need       : {p['need'][:72]}")
            if p.get("error"):
                lines.append(f"     error      : {p['error'][:72]}")
            paths = p.get("artifact_paths") or {}
            if paths.get("code"):
                lines.append(f"     code       : {paths['code']}")
            return "\n".join(lines)

        if event == "tool_registry":
            lines = [
                f"\n[TOOL] cycle={cycle}  {p.get('module')}  "
                f"{p.get('n_tools', 0)} active synthesised tool(s)"
            ]
            for t in p.get("tools") or []:
                lines.append(f"     tool       : {t.get('name')} (source={t.get('source')})")
            return "\n".join(lines)

        if event == "fix":
            # Without this branch the whole M5 result reached the console as a
            # json.dumps of the payload — the one stage whose answer is the
            # point of the run was the one stage nobody could read.
            best = p.get("best") or {}
            attempted = p.get("attempted") or []
            n = len(attempted) if isinstance(attempted, list) else 0
            head = f"\n[M5] {n} repair candidate(s) tried"
            ref = best.get("ref") or best.get("name")
            if ref:
                head += f"; best = {ref}"
            lines = [head]
            if best.get("headline"):
                lines.append(f"     what it does  : {best['headline']}")
            flips = _flip_phrase(best)
            if flips:
                lines.append(f"     result        : it {flips}")
            odds = _odds_phrase(best.get("e_value"))
            if odds:
                strong = "strong enough to count" if best.get("reject") else "not strong enough to count"
                lines.append(f"     is it luck?   : evidence runs {odds} against chance — {strong}")
            independent = int(best.get("n_model_independent") or 0)
            if independent:
                lines.append(
                    f"     careful       : {independent} case"
                    f"{' was' if independent == 1 else 's were'} solved by the added "
                    "code rather than by the model, and were left out of the count"
                )
            lines.append(
                "     verdict       : "
                + ("a repair was confirmed" if p.get("fixed")
                   else "no candidate passed the repair gate")
            )
            if best.get("tier"):
                lines.append(f"     tier={best['tier']}  effect={best.get('effect')}"
                             f"  e_value={best.get('e_value')}  reject={best.get('reject')}")
            return "\n".join(lines)

        if event == "stage_skipped":
            return (f"\n[{str(p.get('stage', '?')).upper()}] cycle={cycle}  skipped"
                    f" ({p.get('reason_code')})"
                    + (f" — {p['detail']}" if p.get("detail") else ""))

        if event == "loop_end":
            stopped_by = p.get("stopped_by")
            if stopped_by is not None:
                return f"\n[DONE] cycles={p.get('cycles')}  stopped_by={stopped_by}"
            return f"\n[DONE] cycles={p.get('cycles')}  resolved={p.get('resolved')}"

        return json.dumps(p, default=str)


class RunLogger:
    """Structured JSONL logger + artifact sink for one :class:`AutoDiagnoseLoop` run.

    Each call to a ``log_*`` method appends one JSON object to ``run_log.jsonl``
    (always including a ``ts`` ISO-8601 timestamp and a ``cycle`` index).
    Heavy artifacts from M1 results are written to ``artifacts/`` as ``.npy``
    (numpy / torch tensors) or ``.json`` (dicts / lists).

    The underlying :attr:`logger` is a standard :class:`logging.Logger` named
    ``evalrx.run.<run_dir_name>``.  It does **not** propagate to the root
    logger so the library stays silent by default.  Attach additional handlers
    to customise where and how events appear::

        rl = RunLogger("runs/exp_01")
        rl.logger.addHandler(logging.StreamHandler(sys.stderr))

    Args:
        run_dir:  Directory to write into.  Created if it does not exist.
                  Defaults to ``runs/<YYYYMMDD_HHMMSS>/`` relative to cwd.
        verbose:  When ``True``, attach a stdout :class:`logging.StreamHandler`
                  with human-readable formatting.  Equivalent to::

                      rl.logger.addHandler(
                          logging.StreamHandler(sys.stdout)
                          # formatted by _VerboseFormatter
                      )

    The logger is safe to use as a context manager::

        with RunLogger("runs/my_exp", verbose=True) as rl:
            loop = AutoDiagnoseLoop(model=model, run_logger=rl)
            loop.run(cases)
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        *,
        verbose: bool = False,
        trace_id: str | None = None,
        context: "Any | None" = None,
        observability_mode: str | None = None,
    ) -> None:
        # When a RunContext is supplied it owns the whole run directory and all
        # of the subdirectory paths; RunLogger simply borrows them.  This keeps a
        # single source of truth for layout while preserving the historical
        # standalone constructor (``RunLogger("runs/exp_01")``) unchanged.
        self._context = context
        if context is not None:
            self.run_dir = context.root
            self.artifact_dir = context.artifacts_dir
            self.experiments_dir = context.experiments_dir
            self.tools_dir = context.tools_dir
            self.workspace_dir = context.workspace_dir
            self.fixes_dir = context.fixes_dir
            self.prompts_dir = context.prompts_dir
            self.log_path = context.log_path
            # Heatmaps/line plots are consolidated under the context's figures/
            # dir rather than living next to the .npy data in artifacts/.
            self._figures_dir: Path | None = context.figures_dir
        else:
            if run_dir is None:
                run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(run_dir)
            self.artifact_dir = self.run_dir / "artifacts"
            # Dedicated, human-navigable sinks for the heavier event payloads.
            #   experiments/  — M5 experiment scripts, run stdout/stderr, the agent's
            #                   intermediate thinking (CLI narration / LLM phase log)
            #   tools/        — code the agent synthesised for new probes / stats tools
            #   workspace/    — per-event snapshots of the sandbox working directory
            self.experiments_dir = self.run_dir / "experiments"
            self.tools_dir = self.run_dir / "tools"
            self.workspace_dir = self.run_dir / "workspace"
            # fixes/ — one self-contained record per tiered-repair attempt, plus an
            #          outcome.md summarising all candidates and the escalation
            #          recommendation.  Written by log_fix().
            self.fixes_dir = self.run_dir / "fixes"
            # prompts/ — the verbatim prompt and raw response of every LLM judge
            #            call (M1 analyzer selection, M2 analysis, M3 diagnosis), so
            #            each conclusion can be traced back to exactly what the judge
            #            was shown and what it returned.
            self.prompts_dir = self.run_dir / "prompts"
            self.log_path = self.run_dir / "run_log.jsonl"
            # Legacy standalone mode: figures land alongside their .npy in artifacts/.
            self._figures_dir = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # The loop stamps this at the top of every cycle so generator-level
        # events (which have no cycle of their own) can be correlated with the
        # M1→M4 events around them.  -1 means "outside any cycle" (e.g. post-loop M5).
        self.current_cycle: int = -1
        # Stable ordering for Langfuse/API readers.  Timestamps alone are not
        # sufficient when a stage emits several records in the same clock tick.
        self._event_seq: int = 0
        # Case records are durable evidence, not renderer-side joins.  Keep the
        # method idempotent because split analysis/confirm workflows can reuse
        # one logger and present the same case more than once.
        self._logged_case_ids: set[str] = set()
        # Monotonic counter so codegen artifacts written in the same cycle never
        # collide on filename. Lock guards the increment in case codegen calls
        # are ever issued from parallel analyzer threads (currently they aren't:
        # ProbeAgent's ThreadPoolExecutor parallelizes analyzer execution only,
        # not codegen — see probe_agent.py).
        self._codegen_seq: int = 0
        self._codegen_lock = threading.Lock()

        # trace_id ties all events from a single AutoDiagnoseLoop.run() call
        # together — including events from any recursive or nested sub-loops.
        # Callers may supply their own ID (e.g. to correlate with an outer
        # pipeline) or let RunLogger generate a fresh UUID.
        self.trace_id: str = trace_id or str(uuid.uuid4())

        # Opt-in, warn-only schema self-check (see _validate_event). Off by
        # default so the common path stays dependency-free and never raises.
        import os
        self._validate_events: bool = bool(os.environ.get("EVALRX_VALIDATE_LOG"))

        self.logger = logging.getLogger(f"evalrx.run.{self.run_dir.name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setFormatter(_JsonFormatter())
        self.logger.addHandler(self._file_handler)

        self._console_handler: logging.StreamHandler | None = None
        if verbose:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setFormatter(_VerboseFormatter())
            self.logger.addHandler(self._console_handler)

        # Dedicated sink for `log_model_call` — every generate/forward/logprobs/
        # chat call an analyzer makes to the TARGET model (see
        # model_instrumentation.InstrumentedModel), as opposed to the judge/coder
        # LLM calls above, which already had verbatim coverage via
        # `_save_judge_io`. Kept in its own file rather than inline in
        # run_log.jsonl: an analyzer like self_consistency or coverage_gap can
        # make dozens of model calls per case, which would drown out the
        # cycle-level narrative in the main log. A standard FileHandler is used
        # for the same reason as `self.logger` above: it is already thread-safe,
        # so concurrent analyzers (ProbeAgent's ThreadPoolExecutor) can log
        # through it without a bespoke lock around file I/O.
        self.model_calls_path = self.run_dir / "model_calls.jsonl"
        self._model_call_logger = logging.getLogger(f"evalrx.model_calls.{self.run_dir.name}")
        self._model_call_logger.setLevel(logging.DEBUG)
        self._model_call_logger.propagate = False
        model_call_handler = logging.FileHandler(self.model_calls_path, encoding="utf-8")
        model_call_handler.setFormatter(_JsonFormatter())
        self._model_call_logger.addHandler(model_call_handler)
        self._model_call_seq = 0
        # Buffered per cycle so `log_probe` can replay each analyzer's calls as
        # properly-nested Langfuse generations once the per-analyzer span
        # exists — that span is only created in `log_probe`, which runs AFTER
        # ProbeAgent.probe() (and therefore every model call in it) completes.
        # The JSONL write above is NOT gated on this: it happens immediately,
        # so the durable record survives even if `log_probe` is never reached
        # (e.g. the process dies mid-cycle).
        self._pending_model_calls: "dict[int, list[dict[str, Any]]]" = {}
        self._pending_lock = threading.Lock()

        # Primary Langfuse & OpenTelemetry Tracing Engine
        from evalrx.observability.tracer import DiagnosticTracer
        self.tracer = DiagnosticTracer(
            run_dir=self.run_dir, mode=observability_mode, auto_sync=True,
        )
        self.tracer.trace_id = self.trace_id

    # ------------------------------------------------------------------
    # Run provenance
    # ------------------------------------------------------------------

    def log_run_start(self, config: "dict[str, Any] | None" = None) -> None:
        """Record a ``run_start`` event with the settings that produced this run.

        *config* is whatever the caller knows (model, protocol, judge/coder
        provider+model, max_cycles, cases…).  This method auto-enriches it with
        the evalrx + Python versions and the current git commit so a run can
        be reproduced from ``run_log.jsonl`` alone.  Always written first.
        """
        import platform

        entry: dict[str, Any] = {"event": "run_start"}
        if config:
            entry.update(config)
        entry.setdefault("python_version", platform.python_version())
        try:
            from evalrx import __version__ as _ver  # type: ignore
            entry.setdefault("evalrx_version", _ver)
        except Exception:  # noqa: BLE001
            pass
        commit = self._git_commit()
        if commit:
            entry.setdefault("git_commit", commit)
        self._log(entry, span_id="run_start")

        # Initialize root Langfuse Trace
        model_name = str(entry.get("model") or "Target Model")
        proto = entry.get("protocol") or {}
        proto_desc = proto.get("description", "") if isinstance(proto, dict) else str(proto)
        bench_name = str(entry.get("benchmark_name") or proto_desc or "Benchmark")
        self.tracer.start_trace(
            model=model_name,
            benchmark=bench_name,
            n_cases=int(entry.get("n_cases", 0) or 0),
            metadata=entry,
        )

    def log_cases(self, cases: "Any", *, split: "str | None" = None) -> None:
        """Persist complete case I/O and media references to JSONL + Langfuse.

        One event per case keeps observations independently queryable and avoids
        a single oversized Langfuse payload.  Media remains path-referenced in
        JSONL; ``media_paths`` makes the tracer upload every existing file as a
        Langfuse Media object with content hashing and deduplication.

        ``split`` names the partition the loop put the case in (``"explore"``,
        ``"confirm"``, ``"test"``). The loop logs every partition, so without
        this a reader cannot tell the cases M1-M3 mined from the ones M4 and
        M5 were measured on -- which is the whole point of splitting them.
        """
        for case in cases:
            case_id = str(getattr(case, "id", "") or "")
            if not case_id or case_id in self._logged_case_ids:
                continue
            if hasattr(case, "to_dict"):
                payload = case.to_dict()
            elif isinstance(case, dict):
                payload = dict(case)
            else:
                payload = {"id": case_id, "value": str(case)}
            # Some user-defined expected/observed objects are not JSON-native.
            payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
            media_paths: list[str] = []
            inputs = getattr(case, "inputs", None)
            for kind in ("image", "audio", "video"):
                value = getattr(inputs, kind, None)
                if not isinstance(value, (str, Path)):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    run_relative = self.run_dir / path
                    path = run_relative if run_relative.is_file() else path.resolve()
                if not path.is_file():
                    continue
                try:
                    media_paths.append(str(path.resolve().relative_to(self.run_dir.resolve())))
                except ValueError:
                    # Langfuse is the durable source of truth, so external case
                    # media must enter the run-owned outbox before its original
                    # path can disappear. Content-prefix naming deduplicates the
                    # common case where several records share one attachment.
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    media_dir = self.artifact_dir / "case_media"
                    media_dir.mkdir(parents=True, exist_ok=True)
                    copied = media_dir / f"{digest.hexdigest()[:16]}_{path.name}"
                    if not copied.exists():
                        shutil.copy2(path, copied)
                    media_paths.append(str(copied.relative_to(self.run_dir)))
            record: dict[str, Any] = {
                "event": "case_record", "case_id": case_id, "case": payload, "media_paths": media_paths,
            }
            if split:
                record["split"] = str(split)
            self._log(record, span_id=f"case.{case_id}")
            self._logged_case_ids.add(case_id)

    def log_report_published(self, envelope: "dict[str, Any]") -> None:
        """Record the cached json-render publication as part of the run audit."""
        generated = envelope.get("generated_by") or {}
        self._log(
            {
                "event": "report_published",
                "report_schema_version": int(envelope.get("schema_version") or 1),
                "catalog_version": str(envelope.get("catalog_version") or ""),
                "json_render_version": str(envelope.get("json_render_version") or ""),
                "source_event_seq": int(envelope.get("source_event_seq") or 0),
                "sha256": str(envelope.get("sha256") or ""),
                "generated_by": generated,
                "report_paths": ["report/report_data.json", "report/report_spec.json"],
            },
            span_id="report.publish",
        )

    @staticmethod
    def _git_commit() -> "str | None":
        """Best-effort current git commit hash (short), or None when unavailable.

        Falls back to the ``EVALRX_GIT_COMMIT`` env var when the ``git`` CLI
        can't be used — notably inside the example Docker images, which install
        no ``git`` and carry no ``.git`` dir, so without this the code-version
        provenance promised above would be silently absent in exactly the
        (containerised) mode the examples are meant to run in.
        """
        import os
        import subprocess
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            commit = out.stdout.strip()
            if commit:
                return commit
        except Exception:  # noqa: BLE001
            pass
        return os.environ.get("EVALRX_GIT_COMMIT") or None

    def _save_judge_io(
        self, stem: str, prompt: "str | None", raw: "str | None"
    ) -> "dict[str, Any] | None":
        """Persist a judge prompt + raw response under ``prompts/``; return summary.

        Returns ``{prompt_path, prompt_chars, raw_path, raw_chars}`` (paths
        relative to the run dir) or ``None`` when neither is present.
        """
        if not prompt and not raw:
            return None
        info: dict[str, Any] = {}
        if prompt:
            p = self._save_text(self.prompts_dir, f"{stem}.prompt.txt", str(prompt))
            if p is not None:
                info["prompt_path"] = p
                info["prompt_chars"] = len(str(prompt))
        if raw:
            p = self._save_text(self.prompts_dir, f"{stem}.response.txt", str(raw))
            if p is not None:
                info["raw_path"] = p
                info["raw_chars"] = len(str(raw))
        return info or None

    def log_model_call(
        self,
        *,
        cycle: int,
        analyzer: str,
        call_index: int,
        method: str,
        inputs: Any,
        kwargs: "dict[str, Any]",
        output: Any,
        duration_sec: float,
        error: "str | None",
        case_id: "str | None" = None,
        batch_case_ids: "list[str] | None" = None,
        n_batch_cases: "int | None" = None,
    ) -> None:
        """Record one TARGET-model call made by an analyzer during M1 probing.

        Called by :class:`~evalrx.eval_agent.model_instrumentation.InstrumentedModel`
        for every ``generate``/``forward``/``logprobs``/``chat`` call — including
        the ones an analyzer makes several times per case (resampling,
        counterfactual regeneration, rollouts) and then reduces to a single
        derived score, which previously left no trace of what the model was
        actually asked or what it actually said. ``case_id`` is a best-effort
        exact-prompt match against the analyzer's batch (``None`` when the
        analyzer rewrote the prompt); ``batch_case_ids`` names the (possibly
        truncated — see ``n_batch_cases`` for the true count) bounded set of
        cases this call could belong to, so an unmatched call is scoped rather
        than untraceable. Capped by the caller (not here) because, unlike
        every other field on this record, it is IDENTICAL across every call
        one analyzer makes in a cycle — inlining it uncapped would repeat a
        whole batch's uuids per call rather than per analyzer.

        Writes immediately to ``model_calls.jsonl`` (durable regardless of
        whether the cycle finishes) and separately buffers the record so
        :meth:`log_probe` can replay it into Langfuse nested under that
        analyzer's probe span — the span doesn't exist yet at call time,
        since ``ProbeAgent.probe()`` (where every model call happens) runs
        to completion before the loop calls ``log_probe``.

        The buffer is keyed by ``cycle`` but ``log_probe`` drains the WHOLE
        buffer regardless of that key (see the drain there) — callers that
        stamp ``current_cycle`` after ``probe()`` has already run (or skip
        ``log_probe`` for a round entirely, e.g. ``VLDiagnoseLoop.run_confirm``
        with ``log=False``) would otherwise leak that round's records forever
        under a cycle number ``log_probe`` never pops.
        """
        record: dict[str, Any] = {
            "event": "model_call",
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "trace_id": self.trace_id,
            "cycle": cycle,
            "analyzer": analyzer,
            "call_index": call_index,
            "method": method,
            "case_id": case_id,
            "batch_case_ids": batch_case_ids or [],
            "n_batch_cases": n_batch_cases if n_batch_cases is not None else len(batch_case_ids or []),
            "inputs": inputs,
            "kwargs": kwargs,
            "output": output,
            "duration_sec": round(duration_sec, 4),
        }
        if error is not None:
            record["error"] = error
        # seq must be assigned under the same lock as the buffer append: this
        # method runs from multiple ThreadPoolExecutor workers at once for the
        # exact high-fan-out analyzers (self_consistency, selfcheck,
        # coverage_gap) this instrumentation targets, and an unlocked
        # read-increment-write of a shared counter drops or duplicates seq
        # numbers under concurrent calls.
        with self._pending_lock:
            self._model_call_seq += 1
            record["seq"] = self._model_call_seq
            self._pending_model_calls.setdefault(cycle, []).append(record)
        self._model_call_logger.info("model_call", extra={"_payload": record})

    # ------------------------------------------------------------------
    # Event hooks — called by AutoDiagnoseLoop at each stage
    # ------------------------------------------------------------------

    def log_probe(
        self,
        cycle: int,
        results: dict[str, "Result"],
        schema: "Any | None" = None,
        *,
        cases: "Any | None" = None,
        judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None,
        duration_sec: "float | None" = None,
        failed_analyzers: "dict[str, str] | None" = None,
    ) -> "list[Path]":
        """M1: log findings (JSON) and persist heavy artifacts to disk.

        Everything M1 produces is captured: the inlined ``findings`` plus, per
        analyzer, the COMPLETE result (``result_paths`` → ``*.result.json`` with
        metadata + summary) and heavy arrays (``artifact_paths`` → ``.npy``/
        ``.json``/figures).  ``failed_analyzers`` records analyzers that were
        selected but errored at runtime (and why), so a selected-but-missing
        analyzer is observable instead of silently absent.  The selection judge
        call is saved under ``prompts/`` via ``judge_io``.

        Returns a list of PNG figure paths that were saved for this cycle
        (attention heatmaps, spatial maps, heatmap-on-image overlays, etc.)
        so callers can forward them to the judge as visual context.
        """
        artifact_paths, overlay_pngs = self._save_probe_artifacts(cycle, results)
        result_paths = self._save_probe_results(cycle, results)
        # Every generate/forward/logprobs/chat call an analyzer made against the
        # target model this cycle (see model_instrumentation.InstrumentedModel).
        # Already durable in model_calls.jsonl by the time this runs; drained
        # here (not just read) so replaying it into Langfuse below happens
        # exactly once and the buffer doesn't grow across cycles.
        #
        # Drains EVERY buffered cycle, not just `cycle` — deliberately, not an
        # oversight. `InstrumentedModel` tags each call with whatever
        # `current_cycle` was live on the RunLogger at call time; a caller that
        # stamps `current_cycle` AFTER `probe()` already ran would leave calls
        # sitting under a cycle key this call's `pop(cycle, [])` would never
        # match, silently dropping them from Langfuse. (`_m4_holdout_pass` used
        # to do exactly this before calling this method with cycle=-1; fixed
        # to stamp -1 before its `probe()` call instead — see loop.py.) Taking
        # everything here is the backstop for that class of bug, current or
        # future, not just insurance against one already-fixed instance.
        #
        # It is NOT airtight against every caller: `VLDiagnoseLoop.run_confirm`
        # calls `_do_m1(..., log=False)`, i.e. probes WITHOUT ever calling
        # `log_probe` to drain that round. Those calls stay correctly durable
        # in model_calls.jsonl (written synchronously in log_model_call,
        # unconditionally) but sit in this buffer until the NEXT `log_probe`
        # call — which then reports them under that later cycle's span and
        # `n_model_calls`, inflating it. Accepted trade-off: a call attributed
        # to the wrong cycle's Langfuse span beats one silently discarded, and
        # cycles are never concurrent, so at most one such round's worth ever
        # accumulates before the next drain absorbs it.
        with self._pending_lock:
            pending_calls = [c for bucket in self._pending_model_calls.values() for c in bucket]
            self._pending_model_calls.clear()
        entry: dict[str, Any] = {
            "event": "probe",
            "cycle": cycle,
            "analyzers": list(results),
            "findings": {name: r.findings for name, r in results.items()},
            "result_paths": result_paths,
            "artifact_paths": artifact_paths,
            "n_model_calls": len(pending_calls),
        }
        if pending_calls:
            entry["model_calls_path"] = str(self.model_calls_path.relative_to(self.run_dir))
        examples = _probe_examples(results, cases)
        if examples:
            entry["examples"] = examples
        if failed_analyzers:
            entry["failed_analyzers"] = dict(failed_analyzers)
        if schema is not None:
            entry["selection_rationale"] = getattr(schema, "rationale", "")
            selected = getattr(schema, "selected_analyzers", None)
            if selected is not None:
                entry["selected_analyzers"] = list(selected)
        judge_io = self._save_judge_io(f"c{cycle}_m1_selection", judge_prompt, judge_raw)
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m1")

        # Collect PNG heatmap paths saved for the .npy arrays, plus any
        # heatmap-on-image overlays.  In context mode figures live under
        # figures/; in legacy mode next to the .npy data.
        png_figures: list[Path] = list(overlay_pngs)
        fig_dir = self._figures_dir or self.artifact_dir
        for rel_npy in artifact_paths.values():
            if not rel_npy.endswith(".npy"):
                continue
            png = fig_dir / (Path(rel_npy).name[: -len(".npy")] + ".png")
            if png.exists():
                png_figures.append(png)

        # Native Langfuse Audit — stage span + one sub-span per probe.  The
        # artifact paths (per-analyzer result JSONs, heavy arrays, rendered
        # figures) are linked in metadata so the trace UI can jump straight
        # to the files; the full findings ride on each probe span's output.
        m1_span = self.tracer.start_span(
            name=f"M1: Multi-Dimensional Checkup (Cycle {cycle})",
            stage="M1",
            input_data={"analyzers": list(results.keys()), "selected_analyzers": entry.get("selected_analyzers", [])},
            metadata={
                "duration_sec": duration_sec,
                "artifacts": {
                    "result_paths": result_paths,
                    "artifact_paths": artifact_paths,
                    "figures": [str(p) for p in png_figures],
                },
            },
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name="M1 Analyzer Selection",
                model="judge",
                prompt=judge_prompt or "",
                completion=judge_raw or "",
                span_id=m1_span,
            )
        for name, r in results.items():
            findings = getattr(r, "findings", {}) or {}
            per_case = findings.get("per_case") or []
            probe_span = self.tracer.start_span(
                name=f"Probe: {name}",
                stage=f"M1_{name}",
                input_data={"probe": name},
                parent_id=m1_span,
                metadata={
                    "n_scored": len(per_case) if per_case else (findings.get("n_cases") or findings.get("n_scored")),
                    "artifacts": {"result_path": result_paths.get(name)},
                },
            )
            # Every target-model call THIS analyzer made, nested under its own
            # probe span so the trace tree shows exactly what the model was
            # asked and what it said at each step — not just the aggregate
            # finding the analyzer reduced those calls to. Langfuse mirroring
            # is capped (see _MAX_MIRRORED_CALLS_PER_ANALYZER below): live mode
            # makes one synchronous HTTP call per generation, and a high-fan-out
            # analyzer (self_consistency n=20, coverage_gap k=10, ×N cases) can
            # make thousands of calls in one cycle. model_calls.jsonl always
            # has the full, uncapped record regardless of this cap.
            calls_for_analyzer = [c for c in pending_calls if c.get("analyzer") == name]
            for call in calls_for_analyzer[: self._MAX_MIRRORED_CALLS_PER_ANALYZER]:
                self.tracer.log_generation(
                    name=f"{name} · {call.get('method')} #{call.get('call_index')}",
                    model="target_model",
                    prompt=call.get("inputs"),
                    completion=call.get("error") or call.get("output"),
                    span_id=probe_span,
                    metadata={"duration_sec": call.get("duration_sec"), "error": call.get("error")},
                )
            self.tracer.end_span(
                probe_span,
                output_data={
                    "findings": findings,
                    "n_model_calls": len(calls_for_analyzer),
                    "n_model_calls_mirrored": min(
                        len(calls_for_analyzer), self._MAX_MIRRORED_CALLS_PER_ANALYZER
                    ),
                },
            )
        # An analyzer that raised mid-run has no Result and so no iteration
        # above — but it may well have made model calls before crashing, and
        # those are exactly the most diagnostically interesting ones (what did
        # the model say right before the failure?). Emit them directly under
        # the M1 span rather than silently excluding them from Langfuse. (Both
        # sides are still subject to _MAX_MIRRORED_CALLS_PER_ANALYZER below —
        # model_calls.jsonl is the only place all of them are guaranteed to be.)
        orphaned = {c["analyzer"] for c in pending_calls} - set(results)
        for name in orphaned:
            failed_span = self.tracer.start_span(
                name=f"Probe: {name} (failed)",
                stage=f"M1_{name}",
                input_data={"probe": name},
                parent_id=m1_span,
                metadata={"note": "analyzer raised before producing a Result"},
            )
            calls_for_analyzer = [c for c in pending_calls if c.get("analyzer") == name]
            for call in calls_for_analyzer[: self._MAX_MIRRORED_CALLS_PER_ANALYZER]:
                self.tracer.log_generation(
                    name=f"{name} · {call.get('method')} #{call.get('call_index')}",
                    model="target_model",
                    prompt=call.get("inputs"),
                    completion=call.get("error") or call.get("output"),
                    span_id=failed_span,
                    metadata={"duration_sec": call.get("duration_sec"), "error": call.get("error")},
                )
            self.tracer.end_span(failed_span, status="failed", output_data={
                "n_model_calls": len(calls_for_analyzer),
                "n_model_calls_mirrored": min(
                    len(calls_for_analyzer), self._MAX_MIRRORED_CALLS_PER_ANALYZER
                ),
            })
        self.tracer.end_span(m1_span, output_data={"n_probes": len(results)})
        return png_figures

    def log_analysis(
        self,
        cycle: int,
        report: "AnalysisReport",
        *,
        duration_sec: "float | None" = None,
    ) -> None:
        """M2: log severity, flagged anomalies, and the narrative sent to M3."""
        entry: dict[str, Any] = {
            "event": "analysis",
            "cycle": cycle,
            "severity": report.severity,
            "n_findings": len(report.findings),
            "findings": [str(f) for f in report.findings],
            "narrative": report.narrative,
            # True when M2 ran descriptively (effect sizes + charts) with the e-BH
            # validity verdict DEFERRED — the analysis phase; the dashboard hides
            # supported/not-supported claims until a confirmatory M2 is logged.
            "descriptive_only": bool(getattr(report, "descriptive_only", False)),
        }
        # Which M2 path produced this report, and — when the LLM path was tried
        # and failed — why. Without these, a judge that never answered logs
        # exactly like a judge that answered and found nothing.
        stats_tool = getattr(report, "stats_tool", None)
        if stats_tool:
            entry["stats_tool"] = stats_tool
        fallback_reason = getattr(report, "llm_fallback_reason", None)
        if fallback_reason:
            entry["llm_fallback_reason"] = fallback_reason
        # StatsAnalysisReport extras (present when VLDiagnoseLoop is used)
        conclusion = getattr(report, "conclusion", None)
        if conclusion:
            entry["conclusion"] = conclusion
        evidence_chain = getattr(report, "evidence_chain", None)
        if evidence_chain:
            entry["evidence_chain"] = list(evidence_chain)
        stats_tool_results = getattr(report, "stats_tool_results", None)
        if stats_tool_results:
            entry["stats_tool_results"] = self._externalize_if_large(
                cycle, "stats_tool_results", list(stats_tool_results)
            )
        visualizations = getattr(report, "visualizations", None)
        if visualizations:
            entry["visualizations"] = list(visualizations)
        # Statistical-tool layer: which tools ran, their verdicts, FDR, figures.
        stats_plan = getattr(report, "stats_plan", None)
        if stats_plan:
            entry["stats_plan"] = self._externalize_if_large(cycle, "stats_plan", stats_plan)
        stats_results = getattr(report, "stats_results", None)
        if stats_results:
            entry["stats_results"] = self._externalize_if_large(
                cycle, "stats_results", [r.to_dict() for r in stats_results]
            )
        corrected = getattr(report, "corrected_rejections", None)
        if corrected:
            entry["corrected_rejections"] = self._externalize_if_large(
                cycle, "corrected_rejections", corrected
            )
        figures = getattr(report, "figures", None)
        if figures:
            entry["figures"] = list(figures)
        # M2 LLM-guided judge I/O (present when StatsAnalysisAgent has a judge).
        judge_io = self._save_judge_io(
            f"c{cycle}_m2_analysis",
            getattr(report, "llm_prompt", None),
            getattr(report, "llm_raw", None),
        )
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m2")

        # Native Langfuse Audit
        def _ext_ptr(field: Any) -> "dict[str, Any] | None":
            """The {path, n_items, bytes} pointer when *field* was externalized."""
            return field if isinstance(field, dict) and "path" in field else None

        m2_span = self.tracer.start_span(
            name=f"M2: Screening & Confirmatory Signals (Cycle {cycle})",
            stage="M2",
            input_data={"severity": report.severity, "n_findings": len(report.findings)},
            metadata={
                "duration_sec": duration_sec,
                "stats_tool": stats_tool,
                "artifacts": {
                    "figures": [str(f) for f in (figures or [])],
                    "stats_results": _ext_ptr(entry.get("stats_results")),
                    "stats_plan": _ext_ptr(entry.get("stats_plan")),
                    "stats_tool_results": _ext_ptr(entry.get("stats_tool_results")),
                    "corrected_rejections": _ext_ptr(entry.get("corrected_rejections")),
                },
            },
        )
        if getattr(report, "llm_prompt", None) or getattr(report, "llm_raw", None):
            self.tracer.log_generation(
                name="M2 Statistical Screening Analysis",
                model="judge",
                prompt=getattr(report, "llm_prompt", "") or "",
                completion=getattr(report, "llm_raw", "") or "",
                span_id=m2_span,
            )
        for s in (getattr(report, "stats_results", None) or []):
            s_dict = s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {})
            sig_name = s_dict.get("config", {}).get("signal") or s_dict.get("tool") or "signal"
            eff = s_dict.get("effect")
            pval = s_dict.get("p_value")
            if eff is not None:
                self.tracer.log_score(
                    name=f"m2_effect_{sig_name}",
                    value=float(eff),
                    comment=f"p={pval}",
                    span_id=m2_span,
                )
        self.tracer.end_span(m2_span, output_data={"conclusion": getattr(report, "conclusion", "")})

    def log_explore(
        self,
        cycle: int,
        report: "Any | None",
        *,
        out_dir: "Path | str | None" = None,
        duration_sec: "float | None" = None,
    ) -> None:
        """In-cycle explore step: log what the free-form EDA produced and where.

        *report* is the explorer's :class:`~evalrx.analysis.explorer.ExploratoryAnalysisReport`
        (or ``None`` when the step failed before producing one). This is a
        DESCRIPTIVE event — the explorer's candidate-signal verdicts are
        in-sample host adjudications and are recorded only as counts; nothing
        here is a confirmatory result. *out_dir* is where the report, tables
        and rendered figures were persisted (``exploratory_report.json``,
        ``tables/``, ``figures/``); the dashboard finds them by path.
        """
        ok = bool(getattr(report, "ok", False)) if report is not None else False
        charts = list(getattr(report, "charts", None) or []) if report is not None else []
        rendered = [
            str(c.get("figure_path")) for c in charts
            if isinstance(c, dict) and c.get("figure_path")
        ]
        tables = getattr(report, "tables", None) or {}
        adjudication = dict(getattr(report, "adjudication", None) or {}) if report is not None else {}
        entry: dict[str, Any] = {
            "event": "explore",
            "cycle": cycle,
            "ok": ok,
            "n_observations": len(getattr(report, "observations", None) or []) if report is not None else 0,
            "n_charts": len(charts),
            "n_charts_rendered": len(rendered),
            "n_tables": len(tables) if isinstance(tables, dict) else len(list(tables or [])),
            "n_candidate_signals": len(getattr(report, "candidate_signals", None) or []) if report is not None else 0,
            "n_hypotheses": len(getattr(report, "hypotheses", None) or []) if report is not None else 0,
            # In-sample host verdict counts (descriptive; the confirmatory M2
            # family is untouched by anything the explorer proposed).
            "adjudication": {
                k: adjudication[k] for k in (
                    "method", "alpha", "split", "n_host_adjudicated", "n_rejected",
                    "n_in_family", "n_descriptive_only",
                ) if k in adjudication
            },
            "observations": [str(o) for o in (getattr(report, "observations", None) or [])[:12]] if report is not None else [],
            "caveats": [str(c) for c in (getattr(report, "caveats", None) or [])[:8]] if report is not None else [],
            "figures": rendered,
            "attempts": int(getattr(report, "attempts", 0) or 0) if report is not None else 0,
        }
        error = str(getattr(report, "error", "") or "") if report is not None else "explorer produced no report"
        if error:
            entry["error"] = error
        if out_dir is not None:
            entry["out_dir"] = str(out_dir)
            report_path = Path(out_dir) / "exploratory_report.json"
            if report_path.exists():
                entry["report_path"] = str(report_path)
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.explore")

        # Native Langfuse Audit — the explore coder-agent trajectory: every
        # attempt's raw CLI output becomes a generation; the synthesized
        # analysis.py and the rendered figures/tables are linked as artifacts.
        exp_span = self.tracer.start_span(
            name=f"Explore: Free-form EDA (Cycle {cycle})",
            stage="EXPLORE",
            input_data={"ok": ok, "attempts": entry.get("attempts", 0)},
            metadata={
                "duration_sec": duration_sec,
                "artifacts": {
                    "out_dir": str(out_dir) if out_dir is not None else None,
                    "report_path": entry.get("report_path"),
                    "figures": rendered,
                },
            },
        )
        if report is not None:
            for i, raw in enumerate(getattr(report, "raw_outputs", None) or []):
                self.tracer.log_generation(
                    name=f"Explore Coder Agent (attempt {i + 1})",
                    model="coder_agent",
                    prompt=None,
                    completion=str(raw),
                    span_id=exp_span,
                )
            if getattr(report, "code", None):
                self.tracer.log_generation(
                    name="Explore Analysis Code (analysis.py)",
                    model="coder_agent",
                    prompt=None,
                    completion=str(report.code),
                    span_id=exp_span,
                )
        self.tracer.end_span(
            exp_span,
            output_data={
                "n_observations": entry.get("n_observations", 0),
                "n_candidate_signals": entry.get("n_candidate_signals", 0),
                "code_path": str(Path(out_dir) / "analysis.py") if out_dir else None,
            },
        )

    def log_diagnosis(
        self,
        cycle: int,
        diag: "DiagnosisResult",
        *,
        duration_sec: "float | None" = None,
        explore_figures: "list[str] | None" = None,
    ) -> None:
        """M3: log raw LLM output, the prompt, and every parsed hypothesis.

        *explore_figures* are the (UNCONFIRMED) explorer chart PNGs M3 was shown,
        recorded for the dashboard; they have no bearing on hypothesis survival.
        """
        entry: dict[str, Any] = {
            "event": "diagnosis",
            "cycle": cycle,
            "model_name": diag.model_name,
            "n_hypotheses": len(diag.hypotheses),
            "hypotheses": [
                {
                    "statement": h.statement,
                    # The reader-facing half of the claim. `_HYPOTHESIS` in
                    # log_schema.py has always declared it and both writers
                    # below omitted it, so the plain sentence the judge was
                    # asked for reached the log only under
                    # `proposed_hypotheses` — and the report UI reads THESE
                    # lists. Every screen therefore showed the technical
                    # statement, and nobody could tell a plain line existed.
                    "plain_statement": h.plain_statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else None,
                    # How M3 says this claim should be verified (the LLM's TEST:
                    # line) — was computed but silently dropped before this fix,
                    # leaving no record of how a hypothesis could be checked.
                    "test_design": h.test_design,
                    # the adversarial critic's verdict travels as provenance
                    # (it annotates, never filters — see _validate_hypotheses)
                    "critic": (h.metadata or {}).get("critic"),
                    "critic_reason": (h.metadata or {}).get("critic_reason"),
                }
                for h in diag.hypotheses
            ],
            "raw_judge_output": diag.raw_judge_output,
            "n_critic_kept": int(getattr(diag, "n_critic_kept", 0) or 0),
            "n_critic_rejected": int(getattr(diag, "n_critic_rejected", 0) or 0),
        }
        critic_raw = getattr(diag, "critic_raw_output", "") or ""
        if critic_raw:
            entry["critic_raw_output"] = critic_raw
            # prompt too: the critic now reads the proposer's context + a label
            # summary, and a reviewer must be able to see what it was judged on.
            critic_io = self._save_judge_io(
                f"c{cycle}_m3_critic",
                getattr(diag, "critic_prompt", "") or None, critic_raw,
            )
            if critic_io:
                entry["critic_io"] = critic_io
        proposed = list(getattr(diag, "proposed_hypotheses", None) or [])
        if proposed:
            entry["proposed_hypotheses"] = [
                {
                    "statement": h.statement,
                    "plain_statement": h.plain_statement,
                    "failure_mode": h.predicted_failure_mode,
                    "test_design": h.test_design,
                }
                for h in proposed
            ]
        review_decisions = list(getattr(diag, "review_decisions", None) or [])
        if review_decisions:
            entry["review"] = {
                "n_kept": sum(d.get("decision") == "keep" for d in review_decisions),
                "n_rejected": sum(d.get("decision") == "reject" for d in review_decisions),
                "decisions": review_decisions,
            }
        # Provenance of the (UNCONFIRMED) explorer mechanism notes M3 was shown.
        # Descriptive only — these never enter M2/M4/fix; logged so the dashboard
        # can tag which explore charts/observations each hypothesis cited.
        referenced = getattr(diag, "referenced_charts", None)
        if referenced:
            entry["referenced_charts"] = list(referenced)
        if getattr(diag, "explore_context_used", False):
            entry["explore_context_used"] = True
        if getattr(diag, "failure_modes_used", False):
            entry["failure_modes_used"] = True
        if explore_figures:
            entry["explore_figures"] = list(explore_figures)
        judge_io = self._save_judge_io(
            f"c{cycle}_m3_diagnosis",
            getattr(diag, "prompt", None),
            diag.raw_judge_output,
        )
        if judge_io:
            entry["judge_io"] = judge_io
        review_io = self._save_judge_io(
            f"c{cycle}_m3_adversarial_review",
            getattr(diag, "review_prompt", None),
            getattr(diag, "review_raw", None),
        )
        if review_io:
            entry["review_io"] = review_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"c{cycle}.m3")

        # Native Langfuse Audit
        m3_span = self.tracer.start_span(
            name=f"M3: Root-Cause Diagnosis (Cycle {cycle})",
            stage="M3",
            input_data={"model_name": diag.model_name, "n_hypotheses": len(diag.hypotheses)},
            metadata={"duration_sec": duration_sec},
        )
        m3_prompt = getattr(diag, "prompt", None) or ""
        if not m3_prompt and judge_io and judge_io.get("prompt_path"):
            # The prompt object wasn't retained — read it back from prompts/.
            try:
                m3_prompt = (self.run_dir / judge_io["prompt_path"]).read_text(encoding="utf-8")
            except Exception:
                m3_prompt = ""
        self.tracer.log_generation(
            name="AI Doctor Diagnostician",
            # diag.model_name is the model being diagnosed, not the model that
            # produced this generation. The configured judge is run metadata.
            model=str(self.tracer.trace_metadata.get("judge") or "diagnosis_judge"),
            prompt=m3_prompt,
            completion=diag.raw_judge_output or "",
            span_id=m3_span,
            metadata={"hypotheses": [h.statement for h in diag.hypotheses]},
        )
        if getattr(diag, "review_prompt", None) or getattr(diag, "review_raw", None):
            self.tracer.log_generation(
                name="M3 Adversarial Evidence Review",
                model=str(self.tracer.trace_metadata.get("judge") or "diagnosis_judge"),
                prompt=getattr(diag, "review_prompt", "") or "",
                completion=getattr(diag, "review_raw", "") or "",
                span_id=m3_span,
                metadata={"decisions": review_decisions},
            )
        self.tracer.end_span(m3_span, output_data={
            "n_hypotheses": len(diag.hypotheses),
            "n_proposed": len(proposed) or len(diag.hypotheses),
            "review": entry.get("review"),
        })

    def log_surgery(
        self,
        cycle: int,
        hypothesis: "Hypothesis",
        iv: "InterventionResult",
        *,
        validation_cases: "Any | None" = None,
        duration_sec: "float | None" = None,
        judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None,
    ) -> None:
        """M5/M4: log intervention outcome for one hypothesis.

        M4 results are distinguished by the presence of ``m4_test_name`` in
        ``iv.evidence``; they get span_id ``c{cycle}.m4`` instead of ``.m5``.
        The M4 protocol-consistency judge call (when a judge was used) is
        saved under ``prompts/`` via ``judge_io`` — same pattern as
        M1/M2/M3, closing the last gap in verbatim judge I/O coverage.
        """
        is_m4 = "m4_test_name" in (iv.evidence or {})
        span_suffix = "m4" if is_m4 else "m5"
        entry: dict[str, Any] = {
            "event": "surgery",
            "cycle": cycle,
            "module": span_suffix,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value,
            "fixed": iv.fixed,
            "confidence_score": iv.confidence_score,
            "evidence_dimensions": iv.evidence_dimensions,
            "evidence": iv.evidence,
            "n_refocused_cases": len(iv.new_data) if iv.new_data else None,
        }
        if is_m4:
            # Store one actual held-out input when it is available.  The M4
            # verdict itself remains aggregate and must never be inferred from
            # this example alone.
            candidates = _iter_cases(validation_cases)
            if candidates:
                snapshots = [_case_snapshot(case) for case in candidates]
                snapshot = next((item for item in snapshots if str(item.get("outcome", "")).lower() == "fail"), snapshots[0])
                entry["validation_examples"] = [{
                    "id": f"m4-{snapshot.get('id')}", "kind": "validation_case",
                    "case_id": snapshot.get("id"), **snapshot,
                    "plain_reading": "This is one case in the independent validation pool. The verdict is determined from the full pool, not this case alone.",
                    "evidence_scope": "one case in the independent validation pool",
                }]
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        slug = re.sub(r"[^a-z0-9]+", "_", hypothesis.statement.lower())[:40].strip("_") or "hyp"
        judge_io = self._save_judge_io(f"c{cycle}_{span_suffix}_{slug}", judge_prompt, judge_raw)
        if judge_io:
            entry["judge_io"] = judge_io
        self._log(entry, span_id=f"c{cycle}.{span_suffix}")

        # Native Langfuse Audit
        stage_title = "M4 Adjudication" if is_m4 else "M5 Intervention"
        surg_span = self.tracer.start_span(
            name=f"{stage_title}: {hypothesis.statement[:60]}",
            stage="M4" if is_m4 else "M5_SURGERY",
            input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
            metadata={"status": iv.status.value, "fixed": iv.fixed, "confidence_score": iv.confidence_score},
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name=f"{stage_title}: Protocol Consistency Judge",
                model="judge",
                prompt=judge_prompt or "",
                completion=judge_raw or "",
                span_id=surg_span,
            )
        if iv.confidence_score is not None:
            self.tracer.log_score(
                name="adjudication_confidence",
                value=float(iv.confidence_score),
                comment=f"status={iv.status.value}, fixed={iv.fixed}",
                span_id=surg_span,
            )
        self.tracer.end_span(surg_span, output_data={"evidence": iv.evidence or {}})

    def log_agent_decision(
        self,
        step: int,
        *,
        action: str,
        params: "dict[str, Any] | None" = None,
        rationale: str = "",
        valid: bool = True,
        repair_attempts: int = 0,
        fallback_used: bool = False,
        judge_prompt: "str | None" = None,
        judge_raw: "str | None" = None,
        duration_sec: "float | None" = None,
    ) -> None:
        """AgenticDiagnoseLoop: log one judge decision turn (chosen tool + why).

        ``valid`` is False when the judge's output could not be parsed even
        after a repair attempt and the host fell back to a deterministic
        next-step heuristic instead (see
        :func:`~evalrx.eval_agent.agentic.actions.decide`).
        """
        entry: dict[str, Any] = {
            "event": "agent_decision",
            "step": step,
            "action": action,
            "params": params or {},
            "rationale": rationale,
            "valid": valid,
            "repair_attempts": repair_attempts,
            "fallback_used": fallback_used,
        }
        judge_io = self._save_judge_io(f"s{step}_agent_decision", judge_prompt, judge_raw)
        if judge_io:
            entry["judge_io"] = judge_io
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"s{step}.decision")

        # Native Langfuse Audit — this was the one stage-level judge call with
        # verbatim prompts/*.txt coverage but no Langfuse span/generation, so
        # the most layered path (AgenticDiagnoseLoop's own dispatch judge) was
        # invisible in the trace tree everywhere else was visible.
        decision_span = self.tracer.start_span(
            name=f"Agent Decision: step {step}",
            stage="AGENT_DECISION",
            input_data={"action": action, "params": params or {}},
            metadata={"valid": valid, "repair_attempts": repair_attempts, "fallback_used": fallback_used},
        )
        if judge_prompt or judge_raw:
            self.tracer.log_generation(
                name="Agent Decision Judge",
                model="judge",
                prompt=judge_prompt or "",
                completion=judge_raw or "",
                span_id=decision_span,
            )
        self.tracer.end_span(decision_span, output_data={"action": action, "rationale": rationale})

    def log_agent_tool(
        self,
        step: int,
        *,
        tool: str,
        ok: bool,
        summary: str = "",
        error: "str | None" = None,
        duration_sec: "float | None" = None,
    ) -> None:
        """AgenticDiagnoseLoop: log the outcome of dispatching one tool call.

        Stage-specific events (``probe``/``analysis``/``diagnosis``/``surgery``)
        are still emitted by the wrapped stage itself, keyed by ``cycle=step``;
        this event records the agentic dispatch layer's own accept/reject
        decision (call caps, unmet preconditions, the stop-gate).
        """
        entry: dict[str, Any] = {
            "event": "agent_tool",
            "step": step,
            "tool": tool,
            "ok": ok,
            "summary": summary,
        }
        if error is not None:
            entry["error"] = error
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 3)
        self._log(entry, span_id=f"s{step}.tool")

    def log_fix(self, outcome: "Any") -> None:
        """Post-loop fix module: log the tiered repair attempt + recommendation.

        *outcome* is a :class:`~evalrx.eval_agent.stages.fix_agent.FixOutcome`;
        its ``to_dict()`` carries every attempted candidate (tier, payload,
        paired-stats verdict, repaired/broken case ids) and the escalation
        recommendation when nothing validated.

        In addition to the lean JSONL ``fix`` event, this writes a
        self-contained human record under ``fixes/``: one
        ``NN_<tier>_<name>/record.md`` per attempted candidate plus a top-level
        ``outcome.md`` summarising all attempts and the recommendation — so each
        repair experiment can be read on its own without parsing the log.
        """
        d = outcome.to_dict()
        # Per-case outputs go to ``outputs.jsonl`` beside each attempt's record
        # (written by _write_fix_records), never inline in the JSONL event —
        # 80 cases x a few KB x N candidates would bloat every reader of the
        # log. The event keeps only the count.
        record = self._write_fix_records(d)
        for a in d.get("attempted") or []:
            outputs = a.pop("outputs", None)
            a["n_outputs"] = len(outputs) if isinstance(outputs, dict) else 0
        best_ref = d.get("best")
        if isinstance(best_ref, dict):
            best = best_ref
        elif isinstance(best_ref, str):
            best = next(
                (a for a in d.get("attempted") or [] if a.get("name") == best_ref), {}
            )
        else:
            best = {}
        entry: dict[str, Any] = {"event": "fix", "cycle": -1, "module": "fix"}
        entry.update(d)
        # FixOutcome serializes ``best`` as a candidate name.  Consumers of the
        # event need the selected candidate's metadata, just as Langfuse does.
        entry["best"] = best
        if record is not None:
            entry["record"] = record
        self._log(entry, span_id="fix")

        # Native Langfuse Audit
        fix_span = self.tracer.start_span(
            name="M5: Targeted Repair & Confirmation",
            stage="M5_FIX",
            input_data={"candidates_evaluated": len(d.get("attempted", []))},
        )
        if best.get("effect") is not None:
            self.tracer.log_score(
                name="repair_net_accuracy_gain",
                value=float(best.get("effect", 0.0)),
                comment=f"cured={best.get('n_fixed')}, broken={best.get('n_broken')}",
                span_id=fix_span,
            )
        if (best.get("payload") or {}).get("prompt_template"):
            self.tracer.log_generation(
                name="Winning Repair Patch",
                model="evalrx_repair",
                prompt="Repair Candidate Search",
                completion=best["payload"]["prompt_template"],
                span_id=fix_span,
            )
        self.tracer.end_span(fix_span, output_data={"selected": best.get("name")})

    def log_stage_skipped(self, stage: str, reason_code: str, *, cycle: int = -1, detail: str = "") -> None:
        """Record an explicit non-error lifecycle decision.

        This is intentionally distinct from ``fix``: an empty repair sweep is
        ambiguous, whereas a blocked evidence gate is useful audit evidence.
        """
        entry = {
            "event": "stage_skipped", "stage": stage, "cycle": cycle,
            "reason_code": reason_code, "detail": detail,
        }
        self._log(entry, span_id=f"{stage.lower()}.skipped")
        span = self.tracer.start_span(
            name=f"{stage}: skipped", stage=stage,
            input_data={"reason_code": reason_code}, metadata={"detail": detail},
        )
        self.tracer.end_span(span, output_data={"reason_code": reason_code}, status="skipped")

    def _write_fix_records(self, d: "dict[str, Any]") -> "str | None":
        """Write per-candidate records + ``fixes/outcome.md``; return the outcome path.

        Each attempt's ``record.md`` (human) + ``result.json`` (machine) land
        in its own *trial* folder — ``Path(a["trial_root"])``, allocated by
        :meth:`~evalrx.eval_agent.run_context.RunContext.new_trial` — so
        the validation record sits next to the code that candidate ran and the
        sandbox it ran in, instead of a flat ``fixes/<slug>/`` re-correlated by
        filename.  Falls back to recomputing that flat slug when a candidate
        carries no trial (no ``RunContext`` in play — legacy standalone
        ``RunLogger``).
        """
        attempts = d.get("attempted") or []

        def _eff(v: "Any") -> "Any":
            return round(v, 4) if isinstance(v, float) else v

        # One record + result per attempted candidate, in its own folder.
        rows: "list[tuple[str, dict[str, Any]]]" = []
        for i, a in enumerate(attempts, start=1):
            tier = a.get("tier", "L?")
            name = a.get("name", "candidate")
            trial_root = a.get("trial_root")
            if trial_root:
                dest_dir = Path(trial_root)
                slug = dest_dir.name
            else:
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", f"{i:02d}_{tier}_{name}").strip("_")
                dest_dir = self.fixes_dir / slug
            rows.append((slug, a))
            num = slug.split("_", 1)[0]
            verdict = a.get("verdict") or ("FIXED" if a.get("fixed") else "did not fix")
            cov = a.get("coverage")
            rates_mode = a.get("noise_model") == "paired_rates"
            lines = [
                f"# Fix attempt {num} — {name}  [{tier}]",
                "",
                f"**Outcome:** {'FIXED' if a.get('fixed') else 'did not fix'} "
                f"(verdict: {verdict})",
                f"**Kind:** {a.get('kind')}    **Source:** {a.get('source')}",
                "",
                "## Validation (paired per-case pass rates vs. unmodified baseline)"
                if rates_mode else
                "## Validation (paired McNemar vs. unmodified baseline)",
                f"- pairs tested (applicable): {a.get('n_pairs')}",
                f"- cases fixed: {a.get('n_fixed')}",
                f"- cases broken: {a.get('n_broken')}",
                f"- coverage of failures: {'—' if cov is None else f'{cov:.0%}'}",
                (f"- unstable cases (baseline flips across its samples; weighed, "
                 f"not dropped): {a.get('n_unstable', 0)}")
                if rates_mode else
                f"- unstable cases dropped (noise): {a.get('n_unstable', 0)}",
                f"- model-independent cases excluded (frozen-model control): "
                f"{a.get('n_model_independent', 0)}",
                f"- effect: {_eff(a.get('effect'))}",
                f"- e-value: {_eff(a.get('e_value'))}",
                f"- statistically significant (rejects H0): {a.get('reject')}",
            ]
            if a.get("n_truncated") is not None:
                lines.append(
                    f"- model calls that hit the decode cap: {a.get('n_truncated')}"
                )
            if a.get("noise_model"):
                lines.append(
                    f"- noise model: {a['noise_model']} "
                    f"(k={a.get('n_baseline_samples', 1)} baseline / "
                    f"{a.get('n_candidate_samples', 1)} candidate samples per case)"
                )
                if a.get("baseline_rate") is not None and a.get("candidate_rate") is not None:
                    lines.append(
                        f"- mean per-case pass rate: baseline {a['baseline_rate']:.3f} -> "
                        f"candidate {a['candidate_rate']:.3f}"
                    )
                if a.get("e_value_regression") is not None:
                    lines.append(
                        f"- e-value for the REVERSE direction (candidate worse): "
                        f"{_eff(a.get('e_value_regression'))}"
                    )
            if a.get("summary"):
                lines.append(f"- summary: {a['summary']}")
            outputs = a.get("outputs")
            if isinstance(outputs, dict) and outputs:
                # One JSON line per case: what the candidate produced, tagged
                # fixed/broken/unchanged so a regression can be read against
                # its actual text (truncated? format slip? wrong?).
                fixed_ids = set(a.get("fixed_cases") or [])
                broken_ids = set(a.get("broken_cases") or [])
                rows_out = []
                for cid, text in outputs.items():
                    status = ("fixed" if cid in fixed_ids
                              else "broken" if cid in broken_ids else "unchanged")
                    rows_out.append(json.dumps(
                        {"case_id": cid, "status": status, "output": text},
                        ensure_ascii=False, default=str))
                self._save_text(dest_dir, "outputs.jsonl", "\n".join(rows_out) + "\n")
                lines.append(f"- per-case outputs: outputs.jsonl ({len(outputs)} cases)")
            lines.append("")
            if a.get("fixed_cases"):
                lines.append("## Cases fixed")
                lines += [f"- {c}" for c in a["fixed_cases"]]
                lines.append("")
            if a.get("broken_cases"):
                lines.append("## Cases broken")
                lines += [f"- {c}" for c in a["broken_cases"]]
                lines.append("")
            lines.append("## What was applied")
            lines.append("```json")
            lines.append(json.dumps(a.get("payload") or {}, indent=2, default=str))
            lines.append("```")
            self._save_text(dest_dir, "record.md", "\n".join(lines))
            lean = {k: v for k, v in a.items() if k != "outputs"}
            lean["n_outputs"] = len(outputs) if isinstance(outputs, dict) else 0
            self._save_text(dest_dir, "result.json", json.dumps(lean, indent=2, default=str))

        # Top-level summary across all attempts.
        fixed = d.get("fixed")
        head = [
            "# Fix outcome",
            "",
            f"**Result:** {'FIXED' if fixed else 'NOT FIXED'}",
            f"**Max tier allowed:** {d.get('max_tier')}",
            f"**Best candidate:** {d.get('best') or '—'}",
        ]
        rec = d.get("recommendation")
        if rec:
            tier = rec.get("recommend_tier")
            if tier and tier == d.get("max_tier"):
                # Same tier as the ceiling = "stay here and do X" (e.g. more
                # failing cases / fewer candidates), not an escalation.
                head.append(
                    f"**Recommendation:** stay within {tier} — {rec.get('reason', '')}"
                )
            elif tier:
                head.append(
                    f"**Recommendation:** escalate to {tier} — {rec.get('reason', '')}"
                )
            else:
                action = rec.get("action", "no fix")
                head.append(
                    f"**Recommendation:** {action} — {rec.get('reason', '')}"
                )
        refine = d.get("refine_signal")
        if refine:
            head.append(f"**Re-diagnose:** {refine.get('message', '')}")
        selection = d.get("selection_attempted") or []
        if selection:
            head += [
                "",
                f"## EXPLORE selection attempts ({len(selection)})",
                "",
                "These results were used only to choose a candidate; they are "
                "not confirmation evidence.",
                "",
                "| # | tier | candidate | verdict | n_fixed | n_broken | effect |",
                "|---|------|-----------|---------|---------|----------|--------|",
            ]
            for i, attempt in enumerate(selection, start=1):
                head.append(
                    f"| {i:02d} | {attempt.get('tier')} | {attempt.get('name')} | "
                    f"{attempt.get('verdict')} | {attempt.get('n_fixed')} | "
                    f"{attempt.get('n_broken')} | {_eff(attempt.get('effect'))} |"
                )
            selected = d.get("selected_on_explore")
            head += ["", f"**Selected on EXPLORE:** {selected or 'none'}"]
        head += ["", f"## Attempts ({len(attempts)})", ""]
        if rows:
            head.append("| # | tier | candidate | verdict | n_fixed | n_broken "
                        "| coverage | effect | sig |")
            head.append("|---|------|-----------|---------|---------|----------"
                        "|----------|--------|-----|")
            for slug, a in rows:
                cov = a.get("coverage")
                cov_s = "—" if cov is None else f"{cov:.0%}"
                head.append(
                    f"| {slug.split('_', 1)[0]} | {a.get('tier')} | {a.get('name')} | "
                    f"{a.get('verdict') or ('fixed' if a.get('fixed') else 'no')} | "
                    f"{a.get('n_fixed')} | {a.get('n_broken')} | {cov_s} | "
                    f"{_eff(a.get('effect'))} | "
                    f"{'yes' if a.get('reject') else 'no'} |"
                )
            head += ["", "Each attempt's full record.md + result.json is in its own "
                     "folder above (`NN_<tier>_<name>/`)."]
        return self._save_text(self.fixes_dir, "outcome.md", "\n".join(head))

    def log_loop_end(
        self,
        report: "AutoDiagnoseReport",
        *,
        tokens_used: "int | None" = None,
        timings: "dict[str, float] | None" = None,
    ) -> None:
        """Final summary entry for the diagnosis loop — does **not** close the log.

        ``loop_end`` marks the end of the M1→M4 diagnosis loop, not the end of
        logging: the post-loop experiments (M5 mechanism verification via
        :meth:`AutoDiagnoseLoop.run_m5`, tiered repair via ``run_fix``) run
        *after* ``loop.run()`` returns and must still be recorded.  The logger's
        lifecycle therefore belongs to whoever created it — use it as a context
        manager or call :meth:`close` explicitly when all work is done.  (Each
        event is flushed to disk as it is written, so an unclosed logger never
        loses data.)

        *report* is an :class:`AutoDiagnoseReport` (all three loops return
        this one unified class — ``VLDiagnoseReport`` is an alias for it).
        The ``hasattr`` checks below stay duck-typed so a hand-built
        ``SimpleNamespace`` with only a subset of fields (e.g. in tests) still
        logs cleanly.

        *tokens_used* and *timings* (per-stage wall-clock totals in seconds)
        record the run's cost/latency profile when the loop supplies them.
        """
        entry: dict[str, Any] = {
            "event": "loop_end",
            "cycles": report.cycles,
        }
        if tokens_used is not None:
            entry["tokens_used"] = tokens_used
        if timings:
            entry["timings_sec"] = {k: round(v, 3) for k, v in timings.items()}
            entry["total_duration_sec"] = round(sum(timings.values()), 3)
        # AutoDiagnoseReport shape
        if hasattr(report, "resolved"):
            entry["resolved"] = report.resolved
            hyps = getattr(report, "final_hypotheses", [])
            entry["n_hypotheses"] = len(hyps)
            entry["final_hypotheses"] = [
                {
                    "statement": h.statement,
                    "plain_statement": h.plain_statement,
                    "failure_mode": h.predicted_failure_mode,
                    "status": h.status.value if h.status else None,
                }
                for h in hyps
            ]
        # VLDiagnoseReport shape
        if hasattr(report, "stopped_by"):
            entry["stopped_by"] = report.stopped_by
            all_hyps = getattr(report, "all_hypotheses", [])
            verified = getattr(report, "verified_hypotheses", [])
            entry["n_hypotheses"] = len(all_hyps)
            entry["n_verified"] = len(verified)
            entry["verified_hypotheses"] = [
                {
                    "statement": tr.hypothesis.statement,
                    "failure_mode": tr.hypothesis.predicted_failure_mode,
                    "status": tr.status.value,
                    "confidence": tr.confidence,
                    "protocol_consistent": tr.is_consistent_with_protocol,
                }
                for tr in verified
            ]
        self._log(entry)
        try:
            bundle_out = self.run_dir / "langfuse_trace.json"
            self.tracer.export_bundle(bundle_out)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Experiment log (M5) + workspace snapshot
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        cycle: int,
        hypothesis: "Hypothesis",
        iv: "InterventionResult",
        *,
        module: str = "m5",
    ) -> None:
        """M5: log the *experiment* the agent wrote and ran to test *hypothesis*.

        Consumes the rich ``iv.experiment`` payload attached by
        :class:`~evalrx.eval_agent.stages.surgery.SurgeryAgent` (the
        generated script(s), the run's stdout/stderr, the verdict, and the
        agent's intermediate thinking — the CLI agent's narration or the
        multi-phase LLM ``validation_log``).

        With a *trial* (``iv.experiment["trial_root"]``, a ``RunContext`` in
        play), everything lands in that one self-contained folder — its live
        ``workspace/`` already holds the sandbox the script ran in (kept on
        success, see ``ExperimentSandbox(cleanup=False)``), so no separate
        snapshot copy is made.  Without one (legacy / no ``RunContext``),
        heavy text is written flat under ``experiments/`` with a ``{stem}_``
        prefix and the sandbox is best-effort copied into
        ``workspace/<stem>/`` — exactly as before.

        Falls back gracefully (logs only the scalar evidence) when
        ``iv.experiment`` is absent, so passive / label-correlation
        interventions still produce an experiment event.
        """
        exp = getattr(iv, "experiment", None) or {}
        prefix = f"c{cycle}" if cycle >= 0 else "post"
        stem = f"{prefix}_{module}"
        trial_root = exp.get("trial_root")
        if trial_root:
            dest_dir = Path(trial_root)
            name_prefix = ""
        else:
            dest_dir = self.experiments_dir
            name_prefix = f"{stem}_"

        # 1. Generated source files (the "changes").
        file_paths: dict[str, str] = {}
        files = exp.get("files") or {}
        if not files and exp.get("code"):
            files = {"main.py": exp["code"]}
        for fname, src in files.items():
            p = self._save_text(dest_dir, f"{name_prefix}{fname}", str(src))
            if p is not None:
                file_paths[fname] = p

        # 2. Output + the agent's intermediate thinking.
        text_paths: dict[str, str] = {}
        for key, suffix in (
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
            ("blueprint", "blueprint.yaml"),
            ("cli_raw_output", "agent_thinking.txt"),
        ):
            val = exp.get(key)
            if val:
                p = self._save_text(dest_dir, f"{name_prefix}{suffix}", str(val))
                if p is not None:
                    text_paths[key] = p
        vlog = exp.get("validation_log")
        if vlog:
            p = self._save_text(
                dest_dir, f"{name_prefix}phase_log.txt",
                "\n".join(str(x) for x in vlog),
            )
            if p is not None:
                text_paths["validation_log"] = p

        # 3. Snapshot the workspace the agent operated in — skipped when it's
        # already durable inside a trial (trial.workspace/ IS the live sandbox).
        workspace = None
        workdir = exp.get("workdir")
        if workdir and not trial_root:
            workspace = self._snapshot_workspace(stem, workdir)

        entry: dict[str, Any] = {
            "event": "experiment",
            "cycle": cycle,
            "module": module,
            "hypothesis": hypothesis.statement,
            "failure_mode": hypothesis.predicted_failure_mode,
            "status": iv.status.value if iv.status else None,
            "fixed": iv.fixed,
            "provider": exp.get("provider"),
            "verdict": exp.get("verdict"),
            "metrics": exp.get("metrics"),
            "returncode": exp.get("returncode"),
            "timed_out": exp.get("timed_out"),
            "cli_usage": exp.get("cli_usage"),
            "llm_calls": exp.get("llm_calls"),
            "sandbox_runs": exp.get("sandbox_runs"),
            "code_paths": file_paths,
            "output_paths": text_paths,
            "workspace_snapshot": workspace,
            "trial_root": trial_root,
        }
        # Human-readable, self-contained record: open this one file to
        # understand the whole experiment without parsing JSONL.
        record = self._write_experiment_record(entry, dest_dir, name_prefix)
        if record is not None:
            entry["record"] = record
        self._log(entry, span_id=f"{prefix}.{module}")

        # Native Langfuse Audit — this stage wrote the coder agent's full
        # trajectory to disk (cli_raw_output → agent_thinking.txt) but never
        # created a span, so the M4 coder run was on disk yet absent from the
        # trace tree (unlike explore's coder agent, logged just below via
        # log_explore's raw_outputs).
        exp_span = self.tracer.start_span(
            name=f"{module.upper()}: Experiment — {hypothesis.statement[:60]}",
            stage=f"{module.upper()}_EXPERIMENT",
            input_data={"hypothesis": hypothesis.statement, "failure_mode": hypothesis.predicted_failure_mode},
            metadata={"provider": exp.get("provider"), "returncode": exp.get("returncode")},
        )
        cli_raw = exp.get("cli_raw_output")
        if cli_raw:
            self.tracer.log_generation(
                name=f"{module.upper()} Coder Agent",
                model=str(exp.get("provider") or "coder_agent"),
                prompt=None,
                completion=str(cli_raw),
                span_id=exp_span,
            )
        vlog = exp.get("validation_log")
        if vlog:
            self.tracer.log_generation(
                name=f"{module.upper()} Validation Log",
                model=str(exp.get("provider") or "coder_agent"),
                prompt=None,
                completion="\n".join(str(x) for x in vlog),
                span_id=exp_span,
            )
        self.tracer.end_span(
            exp_span,
            output_data={"status": entry.get("status"), "fixed": entry.get("fixed"), "verdict": entry.get("verdict")},
        )

    def _write_experiment_record(
        self, entry: "dict[str, Any]", dest_dir: Path, name_prefix: str
    ) -> "str | None":
        """Write a one-page Markdown summary of an M5 experiment.

        Lands in *dest_dir* with *name_prefix* — the same trial folder (no
        prefix) or the flat ``experiments/`` dir (``{stem}_`` prefix) the rest
        of this experiment's files just went to.
        """
        status = (entry.get("status") or "unknown").upper()
        lines = [
            f"# Experiment — {entry.get('module', 'm5').upper()}  ({status})",
            "",
            f"**Hypothesis:** {entry.get('hypothesis', '')}",
            f"**Failure mode:** {entry.get('failure_mode', '—')}",
            "",
            f"**Verdict:** {entry.get('verdict')}    "
            f"**Fixed:** {entry.get('fixed')}",
            "",
        ]
        metrics = entry.get("metrics") or {}
        if metrics:
            lines.append("## Metrics")
            for k, v in metrics.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        lines.append("## How it ran")
        for label, key in (
            ("provider", "provider"), ("return code", "returncode"),
            ("timed out", "timed_out"), ("LLM calls", "llm_calls"),
            ("sandbox runs", "sandbox_runs"),
        ):
            if entry.get(key) is not None:
                lines.append(f"- {label}: {entry[key]}")
        lines.append("")
        files = {**(entry.get("code_paths") or {}), **(entry.get("output_paths") or {})}
        if files:
            lines.append("## Files")
            for name, path in files.items():
                lines.append(f"- `{path}`  — {name}")
            lines.append("")
        return self._save_text(dest_dir, f"{name_prefix}record.md", "\n".join(lines))

    # ------------------------------------------------------------------
    # Tool synthesis — agent generates new probes / stats tools on demand
    # ------------------------------------------------------------------

    def log_tool_codegen(
        self,
        *,
        module: str,
        name: str,
        need: str,
        source: str,
        ok: bool,
        code: str = "",
        prompt: str = "",
        raw_output: str = "",
        raw_stream: str = "",
        error: str = "",
        stdout: str = "",
        cycle: "int | None" = None,
        extra: "dict[str, Any] | None" = None,
    ) -> None:
        """Log one tool-synthesis *attempt* (success OR failure).

        Called from inside a generator (ProbeGenerator / WhiteboxProbeGenerator
        / StatsToolGenerator) the moment it writes code, so the prompt, the raw
        code produced, the backend used (``cli:<provider>`` vs ``llm``) and the
        validation outcome are captured even when the attempt fails to compile
        or run — exactly the cases that vanish today.  The code/prompt/agent
        output are written under ``tools/``; the JSONL event records the paths
        plus the pass/fail outcome.

        ``module`` is e.g. ``"m1_probe"``, ``"m1_whitebox"`` or ``"m2_stats"``.
        """
        cyc = self.current_cycle if cycle is None else cycle
        with self._codegen_lock:
            self._codegen_seq += 1
            seq = self._codegen_seq
        prefix = f"c{cyc}" if cyc >= 0 else "post"
        stem = f"{prefix}_{module}_{name}_{seq:02d}"

        paths: dict[str, str] = {}
        for key, content, suffix in (
            ("code", code, "code.py"),
            ("prompt", prompt, "prompt.txt"),
            ("raw_output", raw_output, "agent_thinking.txt"),
            ("raw_stream", raw_stream, "agent_raw_stream.txt"),
            ("stdout", stdout, "stdout.txt"),
        ):
            if content:
                p = self._save_text(self.tools_dir, f"{stem}_{suffix}", str(content))
                if p is not None:
                    paths[key] = p

        entry: dict[str, Any] = {
            "event": "tool_codegen",
            "cycle": cyc,
            "module": module,
            "tool_name": name,
            "need": need,
            "source": source,
            "ok": ok,
            "error": error or None,
            "artifact_paths": paths,
        }
        if extra:
            entry.update(extra)
        self._log(entry, span_id=f"{prefix}.{module}.codegen")

        # Native Langfuse Audit — the coder agent's trajectory for this
        # synthesis attempt: prompt, raw agent output, generated code, and
        # the tools/ paths where they were persisted.
        cg_span = self.tracer.start_span(
            name=f"Tool Codegen: {module}/{name}",
            stage=f"CODEGEN_{module}",
            input_data={"need": need, "source": source},
            metadata={"ok": ok, "error": error or None, "artifacts": paths},
        )
        if prompt or raw_output:
            self.tracer.log_generation(
                name=f"Tool Synthesis: {name}",
                model=source,
                prompt=prompt or "",
                completion=raw_stream or raw_output or code,
                span_id=cg_span,
            )
        self.tracer.end_span(cg_span, output_data={"ok": ok, "code_chars": len(code or "")})

    def log_tool_registry(
        self,
        cycle: int,
        module: str,
        generated: "list[Any]",
    ) -> None:
        """Snapshot which synthesised tools are registered/active for *cycle*.

        *generated* is a list of objects carrying ``name``/``code``/``need``/
        ``source`` attributes (``GeneratedProbe`` / ``GeneratedStatsTool``).
        Records the active tool registry for the cycle and persists each tool's
        source under ``tools/`` (idempotent by name).
        """
        if not generated:
            return
        prefix = f"c{cycle}" if cycle >= 0 else "post"
        tools: list[dict[str, Any]] = []
        for g in generated:
            name = getattr(g, "name", "tool")
            code = getattr(g, "code", "")
            code_path = None
            if code:
                code_path = self._save_text(
                    self.tools_dir, f"{module}_{name}.code.py", str(code)
                )
            tools.append({
                "name": name,
                "need": getattr(g, "need", ""),
                "source": getattr(g, "source", ""),
                "code_path": code_path,
            })
        self._log(
            {
                "event": "tool_registry",
                "cycle": cycle,
                "module": module,
                "n_tools": len(tools),
                "tools": tools,
            },
            span_id=f"{prefix}.{module}.registry",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all log handlers and persist Langfuse trace bundle."""
        try:
            bundle_out = self.run_dir / "langfuse_trace.json"
            self.tracer.export_bundle(bundle_out)
        except Exception:
            pass
        # Events are children of the root Langfuse chain.  Drain them before
        # ending that chain so the remote trace keeps its native hierarchy.
        self.tracer.flush()
        self.tracer.end_trace({
            "spans": len(self.tracer.spans),
            "generations": len(self.tracer.generations),
            "scores": len(self.tracer.scores),
        })
        self.tracer.flush()
        for handler in (self._file_handler, self._console_handler):
            if handler is not None:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)
        for handler in list(self._model_call_logger.handlers):
            handler.flush()
            handler.close()
            self._model_call_logger.removeHandler(handler)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RunLogger(run_dir={str(self.run_dir)!r})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, entry: dict[str, Any], *, span_id: str | None = None) -> None:
        self._event_seq += 1
        entry["schema_version"] = RUN_LOG_SCHEMA_VERSION
        entry["ts"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        entry["trace_id"] = self.trace_id
        entry["event_seq"] = self._event_seq
        if span_id is not None:
            entry["span_id"] = span_id
        if self._validate_events:
            self._validate_event(entry)
        self.logger.info("run_event", extra={"_payload": entry})
        try:
            self.tracer.record_event(entry, event_seq=self._event_seq)
        except Exception as exc:  # noqa: BLE001 - a telemetry disk error must not lose a run
            warnings.warn(f"RunLogger: could not queue Langfuse event: {exc}")

    def _validate_event(self, entry: dict[str, Any]) -> None:
        """Opt-in self-check: warn (never raise) when an event violates the schema.

        Enabled by ``EVALRX_VALIDATE_LOG`` (see ``__init__``).  Kept warn-only
        and fully guarded so turning it on can never break a run — it's a
        developer/CI aid to catch a producer drifting from the published schema,
        not a runtime gate.  Needs the optional ``jsonschema`` dep; a missing dep
        or any other hiccup degrades silently to "not validated".
        """
        try:
            from evalrx.eval_agent.log_schema import validate_event
            validate_event(entry)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 — never let validation break logging
            warnings.warn(
                f"RunLogger: event {entry.get('event')!r} violates run_log schema: {exc}"
            )

    def _save_text(self, directory: Path, stem: str, text: str) -> "str | None":
        """Write *text* to ``directory/stem`` (creating *directory*); return rel path.

        ``stem`` already carries the extension (e.g. ``c0_m5_main.py``).  Returns
        the path relative to :attr:`run_dir` for embedding in the JSONL event, or
        ``None`` if writing fails.
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / stem
            path.write_text(text, encoding="utf-8")
            return str(path.relative_to(self.run_dir))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: could not save text artifact {stem!r}: {exc}")
            return None

    def save_artifact_json(self, stem: str, obj: Any) -> "str | None":
        """Write *obj* as JSON to ``artifacts/<stem>``; return the run-relative path.

        A conventional-file sibling to the JSONL log for artifacts a downstream
        reader (e.g. the dashboard) looks up by fixed name rather than by
        scanning events — e.g. ``failure_modes.json``. A repeat call with the
        same *stem* overwrites (latest wins); callers that need history should
        vary the stem themselves.
        """
        return self._save_text(
            self.artifact_dir, stem, json.dumps(obj, indent=2, default=str)
        )

    # Above this size, M2 stats payloads are externalized like every other
    # heavy field (judge I/O, M1 artifacts) instead of inlined in the JSONL
    # line — typical runs stay well under this, so the common case is
    # unaffected and still jq/tail -f friendly.
    _INLINE_MAX_BYTES = 4096

    # Caps how many of one analyzer's model_call records get mirrored into
    # Langfuse per log_probe() (see log_probe). model_calls.jsonl always keeps
    # every call regardless — this only bounds the live-mode Langfuse client,
    # which makes one synchronous HTTP request per generation and would
    # otherwise turn a high-fan-out analyzer's cycle (self_consistency n=20,
    # coverage_gap k=10, × every case in the batch) into thousands of
    # sequential network calls.
    _MAX_MIRRORED_CALLS_PER_ANALYZER = 50

    def _externalize_if_large(
        self, cycle: int, key: str, value: Any, *, threshold_bytes: int = _INLINE_MAX_BYTES,
    ) -> Any:
        """Inline *value* unless its JSON size exceeds *threshold_bytes*.

        Oversized values are persisted under ``artifacts/`` and replaced with
        ``{"path", "n_items", "bytes"}`` so the JSONL line stays lean.
        """
        serialized = json.dumps(value, default=str)
        size = len(serialized.encode("utf-8"))
        if size <= threshold_bytes:
            return value
        prefix = f"c{cycle}" if cycle >= 0 else "post"
        path = self._save_text(self.artifact_dir, f"{prefix}_m2_{key}.json", serialized)
        summary: dict[str, Any] = {"bytes": size}
        if isinstance(value, (list, dict)):
            summary["n_items"] = len(value)
        if path is not None:
            summary["path"] = path
        return summary

    # Files worth keeping in a workspace snapshot — code, data, prose, logs.
    # Heavy binaries (weights, tensors, images) are skipped to keep snapshots
    # small; the .npy/.png analyzer artifacts are already saved under artifacts/.
    _SNAPSHOT_SUFFIXES = frozenset(
        {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".log", ".toml"}
    )
    _SNAPSHOT_MAX_BYTES = 2_000_000  # skip any single file larger than 2 MB

    def _snapshot_workspace(self, stem: str, workdir: "str | Path") -> "dict[str, Any] | None":
        """Copy text/code/data files from *workdir* into ``workspace/<stem>/``.

        Returns a manifest ``{"dir": <rel path>, "files": [...], "skipped": n}``
        or ``None`` when *workdir* does not exist.  The sandbox deletes its
        working directory on success, so this is best-effort: callers should
        snapshot promptly after the run.
        """
        import shutil

        src = Path(workdir)
        if not src.exists() or not src.is_dir():
            return None
        dest = self.workspace_dir / stem
        kept: list[str] = []
        skipped = 0
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in self._SNAPSHOT_SUFFIXES:
                    skipped += 1
                    continue
                try:
                    if f.stat().st_size > self._SNAPSHOT_MAX_BYTES:
                        skipped += 1
                        continue
                    rel = f.relative_to(src)
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)
                    kept.append(str(rel))
                except Exception:  # noqa: BLE001
                    skipped += 1
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: workspace snapshot failed for {stem!r}: {exc}")
            return None
        return {
            "dir": str(dest.relative_to(self.run_dir)),
            "files": kept,
            "skipped": skipped,
        }

    def _save_probe_artifacts(
        self,
        cycle: int,
        results: dict[str, "Result"],
    ) -> "tuple[dict[str, str], list[Path]]":
        """Persist heavy artifacts from all M1 results.

        Returns ``({key: path}, overlay_pngs)``. *overlay_pngs* are heatmap-on-
        image visualisations from ``Result`` subclasses defining an
        ``image_overlays()`` hook (duck-typed — e.g. ``RelativeAttentionResult``),
        saved alongside the bare heatmaps so a multimodal judge sees the actual
        photo under the highlighted patches instead of an abstract colour grid.
        """
        paths: dict[str, str] = {}
        overlay_pngs: list[Path] = []
        fig_dir = self._figures_dir or self.artifact_dir
        for analyzer_name, result in results.items():
            for art_name, artifact in result.artifacts.items():
                stem = f"c{cycle}_{analyzer_name}_{art_name}"
                path = self._save_artifact(stem, artifact)
                if path is not None:
                    paths[f"{analyzer_name}/{art_name}"] = str(path.relative_to(self.run_dir))
            image_overlays = getattr(result, "image_overlays", None)
            if image_overlays is not None:
                try:
                    overlay_pngs.extend(image_overlays(fig_dir, f"c{cycle}_{analyzer_name}"))
                except Exception as exc:  # noqa: BLE001 - viz must never break the probe
                    warnings.warn(f"RunLogger: image_overlays failed for {analyzer_name}: {exc}")
        return paths, overlay_pngs

    def _save_probe_results(
        self,
        cycle: int,
        results: dict[str, "Result"],
    ) -> "dict[str, str]":
        """Persist each analyzer's COMPLETE result so M1's full output is
        observable, not just the ``findings`` inlined into the probe event.

        Writes ``artifacts/c{cycle}_{analyzer}.result.json`` carrying the full
        :meth:`Result.to_dict` (findings + metadata + n_cases) plus the rendered
        ``summary()`` text.  Heavy arrays/tensors already go through
        :meth:`_save_probe_artifacts`; this captures everything else.
        """
        paths: dict[str, str] = {}
        for analyzer_name, result in results.items():
            to_dict = getattr(result, "to_dict", None)
            if not callable(to_dict):
                continue  # minimal/duck-typed result with no serialisable view
            try:
                doc = to_dict()
                summary = getattr(result, "summary", None)
                if callable(summary):
                    doc["summary"] = summary()
                doc["artifact_names"] = sorted((getattr(result, "artifacts", None) or {}).keys())
            except Exception as exc:  # noqa: BLE001 - logging must never break M1
                warnings.warn(f"RunLogger: could not serialise result {analyzer_name!r}: {exc}")
                continue
            path = self.artifact_dir / f"c{cycle}_{analyzer_name}.result.json"
            try:
                path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"RunLogger: could not write result {analyzer_name!r}: {exc}")
                continue
            paths[analyzer_name] = str(path.relative_to(self.run_dir))
        return paths

    def _save_artifact(self, stem: str, artifact: Any) -> Path | None:
        """Write one artifact to ``artifacts/<stem>.<ext>``; return path or None.

        For numeric artifacts (tensors, arrays, list-of-tensors):
          - Saves raw data as ``<stem>.npy``.
          - Also saves a ``<stem>.png`` figure when the shape and stem keyword
            are recognised (attention → heatmap, entropy/rollout → line/heatmap).
            Silently skipped when matplotlib is not installed.

        For dict/list artifacts: saves ``<stem>.json``.
        """
        try:
            import numpy as np

            arr = _artifact_to_numpy(artifact)
            if arr is not None:
                path = self.artifact_dir / f"{stem}.npy"
                np.save(path, arr)
                fig_dir = self._figures_dir or self.artifact_dir
                fig_dir.mkdir(parents=True, exist_ok=True)
                _save_artifact_figure(fig_dir, stem, arr)
                return path
            if isinstance(artifact, (dict, list)):
                path = self.artifact_dir / f"{stem}.json"
                path.write_text(json.dumps(artifact, default=str), encoding="utf-8")
                return path
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"RunLogger: could not save artifact {stem!r}: {exc}")
        return None
