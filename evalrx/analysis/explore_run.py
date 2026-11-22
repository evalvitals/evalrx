"""Single-shot exploratory analysis run (the ``evalrx explore`` entry).

One call = one pipeline = one artifact directory. This is the non-interactive
replacement for the retired chat REPL: the explorer writes EDA code, the host
adjudicates any host-checkable candidate statistics, charts are rendered
deterministically, and everything is persisted, ending in a static ``report.html``.

Flow::

    ExploratoryAnalysisAgent.explore_path        # M2: free-form EDA (a CLI coding agent)
      -> adjudicate_report              # host recomputes verdicts (in-sample)
      -> HypothesisAgent.propose        # M3: falsifiable hypotheses from M2's takeaways
      -> render_chart_specs             # spec + CSV -> PNG, host-side
      -> write_report_artifacts         # report.json + figures/ + tables/ + code
      -> write_report                   # report.html (opened when asked)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evalrx.analysis.api import explore
from evalrx.analysis.explorer import (
    RECORDS_FILENAME,
    load_records_from_path,
    truncation_warning,
)
from evalrx.analysis.holdout import holdout_confirm as run_holdout_confirm
from evalrx.analysis.holdout import split_records
from evalrx.viz.renderer import render_chart_specs

# Appended to the analysis question when a held-out confirm phase will follow:
# recipes must be frozen and threshold-explicit or there is nothing to re-test.
HOLDOUT_QUESTION_SUFFIX = (
    " This is the EXPLORATION HALF of a held-out design: a disjoint validate "
    "split exists and will re-test whatever you propose. Make every candidate "
    "signal a FROZEN, threshold-explicit recipe (explicit numeric thresholds, "
    "no re-fitting) so it can be re-evaluated verbatim on the held-out half, "
    "and propose hypotheses precise enough to be graded against held-out "
    "statistics."
)


def _build_judge(model: str):
    """Claude judge for the held-out confirm phase (separate so tests can stub it)."""
    from evalrx.agent_runtime.judges import ClaudeModel

    return ClaudeModel(model=model, effort="low")


_CLAUDE_BACKENDS = ("claude", "claude_code")
_DEFAULT_CLAUDE_JUDGE = "claude-opus-4-8"


def resolve_judge(
    judge_backend: str,
    judge_model: str,
    *,
    coder_provider: str,
    coder_model: str = "",
    coder_binary: str = "",
    timeout_sec: int = 120,
) -> "tuple[Any, dict[str, Any]]":
    """The judge that grades hypotheses on held-out rows, and a record of it.

    *judge_backend* ``"same"`` follows the backend the explorer ran on, so a
    run on a backend other than Claude does not need the Claude CLI as well.
    Any other value names a backend outright. The returned record says which
    backend and model graded, and whether that is the model that proposed the
    hypotheses — a review by the proposer is not an independent check, and the
    report says so rather than leaving it to be assumed.
    """
    backend = coder_provider if judge_backend in ("", "same") else judge_backend
    if backend in _CLAUDE_BACKENDS:
        model = judge_model or _DEFAULT_CLAUDE_JUDGE
        judge: Any = _build_judge(model)
        meta: dict[str, Any] = {"backend": "claude", "model": model, "effort": "low"}
    else:
        from evalrx.agent_runtime.cli_types import CliAgentConfig
        from evalrx.agent_runtime.judges.cli_agent import CliAgentJudge

        same_backend = backend == coder_provider
        model = judge_model or (coder_model if same_backend else "")
        judge = CliAgentJudge(
            CliAgentConfig(
                provider=backend, model=model,
                binary_path=coder_binary if same_backend else "",
                timeout_sec=timeout_sec,
            ),
            timeout_sec=timeout_sec,
        )
        meta = {"backend": backend, "model": model or "(backend default)"}

    proposer_claude = coder_provider in _CLAUDE_BACKENDS
    same_family = (backend in _CLAUDE_BACKENDS) == proposer_claude and (
        proposer_claude or backend == coder_provider
    )
    if same_family:
        reviewer_model = "" if meta["model"] == "(backend default)" else meta["model"]
        if coder_model and reviewer_model and coder_model != reviewer_model:
            meta["same_as_proposer"] = "no"
        elif coder_model == reviewer_model:
            meta["same_as_proposer"] = "yes"
        else:
            # One side runs on its backend's default model, which is not known here.
            meta["same_as_proposer"] = "same_backend"
    else:
        meta["same_as_proposer"] = "no"
    return judge, meta


def same_reviewer_caveat(judge_meta: "dict[str, Any] | None", coder_provider: str) -> str:
    """A sentence for the report when the reviewer is, or may be, the proposer."""
    same = (judge_meta or {}).get("same_as_proposer")
    if same == "yes":
        return (
            f"the hypotheses were proposed and graded by the same model "
            f"({judge_meta.get('model') if judge_meta else ''} on {coder_provider}); "
            "its grades are not an independent review. Signal verdicts and "
            "host-checked predictions do not depend on the judge."
        )
    if same == "same_backend":
        return (
            f"the hypotheses were proposed and graded on the same backend "
            f"({coder_provider}), and at least one of the two used its default "
            "model, so the grader may be the proposer; pass --judge-backend or "
            "--judge-model for an independent review. Signal verdicts and "
            "host-checked predictions do not depend on the judge."
        )
    return ""


#: The four things a reader needs to know about a table before trusting any
#: pattern found in it. Keys are the ``data_card`` fields.
DATA_CARD_QUESTIONS = {
    "unit": "what one row is",
    "extraction": "how a row was cut from the raw object",
    "threshold": "how the outcome or its threshold was set",
    "aggregation": "how results are combined when the system is used",
}


def load_data_card(path: "str | Path") -> dict[str, str]:
    """Read a data card (JSON or YAML mapping) and keep its known fields."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"data card {path} must be a mapping of field to answer")
    return {k: str(v).strip() for k, v in raw.items()
            if k in DATA_CARD_QUESTIONS and str(v).strip()}


def data_card_caveats(card: "dict[str, str] | None") -> list[str]:
    """The data card as report caveats: each answer, then what was left unanswered."""
    card = card or {}
    out = [f"data card — {DATA_CARD_QUESTIONS[k]}: {card[k]}"
           for k in DATA_CARD_QUESTIONS if card.get(k)]
    missing = [DATA_CARD_QUESTIONS[k] for k in DATA_CARD_QUESTIONS if not card.get(k)]
    if missing:
        out.append(
            "data card — not stated: " + "; ".join(missing) + ". Patterns below "
            "may come from how the table was built rather than from the system; "
            "answer these with --data-card or the --row-* flags."
        )
    return out


def run_explore(
    path: str | Path,
    *,
    question: str = "Explore this dataset and surface the patterns that matter.",
    out: str | Path = "evalrx_explore_output",
    coder_provider: str = "antigravity",
    coder_model: str = "",
    coder_binary: str = "",
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
    timeout_sec: int = 120,
    max_attempts: int = 2,
    open_report: bool = False,
    skills: "list[str] | tuple[str, ...]" = (),
    allow_skills: bool = False,
    use_bundled_skills: bool = True,
    outcome_col: str | None = None,
    propose_hypotheses: bool = True,
    holdout_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_confirm: bool = False,
    judge_model: str = "",
    progress_sink: Any = None,
    group_col: str | None = None,
    time_col: str | None = None,
    text_col: str | None = None,
    data_card: "dict[str, str] | None" = None,
    judge_backend: str = "same",
    keep_sandbox: bool = False,
) -> int:
    """Run one exploratory analysis and persist its artifacts.

    Returns a process exit code: ``0`` if the exploration succeeded, else ``1``.
    *open_report* opens the written ``report.html`` in a browser.

    Skills style the agent-authored figures (e.g. the bundled eval-chart-style
    and nature-figure skills). By default (*use_bundled_skills*) the package's
    bundled skills are applied on the claude/agy/codex backends; *skills* adds
    more dirs and *allow_skills* also enables globally-installed
    (`~/.claude/skills`) skills.

    *outcome_col* optionally names the target/label column explicitly (M1
    passes ``"label"``); omit it to let the agent auto-detect an outcome by
    name heuristics, or fall back to unsupervised EDA when there is none.

    *propose_hypotheses* runs M3 (``HypothesisAgent``) on M2's takeaways after
    a successful explore, using the same coding-agent backend; set False to
    skip it (e.g. to save the extra LLM call).

    *holdout_frac* > 0 splits the loaded records BEFORE exploration
    (deterministic, outcome-stratified; the explorer only sees the explore
    share). With *holdout_confirm* the held-out rows then re-test the frozen
    recipes + hypotheses (``analysis.holdout``) and ``confirm_report.json``
    lands next to the exploratory report. Without it the rows are simply
    reserved on disk.

    *group_col* names the unit rows were cut from. The split then keeps each
    group on one side and the held-out intervals resample whole groups. When
    it is not given and a column looks like such a unit, the run says so and
    reports how many held-out rows share a group with an explored row.
    *time_col* names a time column. *text_col* names a free-text column: the
    words over-represented on either side of the outcome in the exploration
    half are frozen as keyword recipes and re-tested with everything else.

    *data_card* answers :data:`DATA_CARD_QUESTIONS`; the answers, and the
    questions left unanswered, are printed in the report's caveats.

    *judge_backend* picks who grades the hypotheses: ``"same"`` (default) uses
    the explorer's backend, anything else names a backend. *judge_model*
    overrides its model.
    """
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source: Any = path
    holdout_rows: "list[dict[str, Any]] | None" = None
    host_caveats: list[str] = data_card_caveats(data_card)
    extra_candidates: list[Any] = []
    split_extra: dict[str, Any] = {}
    if data_card:
        question = question.rstrip() + " DATA CARD (from the user): " + "; ".join(
            f"{DATA_CARD_QUESTIONS[k]} — {v}" for k, v in data_card.items()
            if k in DATA_CARD_QUESTIONS
        ) + "."
    if holdout_frac and holdout_frac > 0:
        load_info: dict[str, Any] = {}
        records = load_records_from_path(
            path, max_rows=max_rows, max_files=max_files,
            include_tool_calls=include_tool_calls, info=load_info,
        )
        warning = truncation_warning(load_info, max_rows)
        if warning:
            print(f"WARNING: {warning}")
            host_caveats.append(warning)
        explore_share, holdout_rows = split_records(
            records, holdout_frac, seed=holdout_seed, outcome_col=outcome_col,
            group_col=group_col,
        )
        if holdout_rows is None:
            print(f"holdout: split impossible (n={len(records)}, frac={holdout_frac}) "
                  "— falling back to a plain in-sample run")
        else:
            source = explore_share
            print(f"holdout: exploring {len(explore_share)} rows, "
                  f"{len(holdout_rows)} held out "
                  + ("(verification follows)" if holdout_confirm
                     else "(reserved, not verified in this mode)"))
            if holdout_confirm:
                question = question.rstrip() + HOLDOUT_QUESTION_SUFFIX
            split_extra, group_caveats = _group_checks(
                records, explore_share, holdout_rows,
                outcome_col=outcome_col, group_col=group_col, time_col=time_col,
            )
            for caveat in group_caveats:
                print(f"WARNING: {caveat}")
            host_caveats.extend(group_caveats)
        if text_col:
            extra_candidates = _keyword_candidates(
                explore_share, text_col, outcome_col or "label"
            )
            if extra_candidates:
                print(f"keywords: froze {len(extra_candidates)} keyword recipe(s) "
                      f"from {text_col!r} on the exploration rows")
    elif text_col:
        print("keywords: --text-col needs --holdout-frac > 0 (keyword recipes are "
              "chosen on one half and tested on the other); ignored")

    result = explore(
        source,
        question=question,
        out=out_dir,
        provider=coder_provider,
        model=coder_model,
        binary=coder_binary,
        outcome_col=outcome_col,
        group_col=group_col,
        time_col=time_col,
        propose_hypotheses=propose_hypotheses,
        skills=skills,
        allow_skills=allow_skills,
        use_bundled_skills=use_bundled_skills,
        max_rows=max_rows,
        max_files=max_files,
        include_tool_calls=include_tool_calls,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        progress_sink=progress_sink,
        keep_sandbox=keep_sandbox,
        caveats=host_caveats,
        extra_candidates=extra_candidates,
    )
    report = result.report

    print(f"ok: {report.ok}")
    print(f"attempts: {report.attempts}")
    print(f"rows loaded: {report.data_profile.get('loaded_rows', report.data_profile.get('n_rows'))}")
    print(f"output: {out_dir}")
    if report.error:
        print(f"error: {report.error}")
    if report.hypotheses:
        print("\nhypotheses (M3, proposed only — not validated):")
        for h in report.hypotheses[:8]:
            print(f"- {h.get('statement')}")
    if report.observations:
        print("\nobservations:")
        for obs in report.observations[:8]:
            print(f"- {obs}")
    if report.candidate_signals:
        print("\ncandidate signals:")
        for signal in report.candidate_signals[:8]:
            print(f"- {signal.name}: {signal.rationale}{_verdict_suffix(signal)}")
    if report.adjudication:
        adj = report.adjudication
        print(
            f"\nhost adjudication ({adj.get('split')}): "
            f"{adj.get('n_rejected', 0)}/{adj.get('n_host_adjudicated', 0)} reject "
            f"(correction: {adj.get('method')}, e-BH family n={adj.get('n_in_family', 0)}, "
            f"BH family n={adj.get('n_in_p_family', 0)}, alpha={adj.get('alpha')})"
        )

    if holdout_rows is not None:
        # Audit trail either way: the exact rows the explorer never saw.
        (out_dir / "holdout_records.json").write_text(
            json.dumps(holdout_rows, default=str), encoding="utf-8"
        )
        split_meta = {
            "n_explore": len(source) if isinstance(source, list) else None,
            "n_holdout": len(holdout_rows),
            "holdout_frac": holdout_frac,
            "seed": holdout_seed,
            "stratified_by": outcome_col or "label",
            "group_col": group_col,
            **split_extra,
        }
        if holdout_confirm:
            judge = None
            judge_meta = None
            confirm_caveats = list(host_caveats)
            try:
                judge, judge_meta = resolve_judge(
                    judge_backend, judge_model, coder_provider=coder_provider,
                    coder_model=coder_model, coder_binary=coder_binary,
                    timeout_sec=timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001 — verdicts degrade to not_judged
                print(f"judge unavailable ({exc}) — hypotheses will be not_judged")
            reviewer = same_reviewer_caveat(judge_meta, coder_provider)
            if reviewer and report.hypotheses:
                print(f"WARNING: {reviewer}")
                confirm_caveats.append(reviewer)
            if progress_sink is not None:
                progress_sink.emit("m4", "started", "Verifying hypotheses on the held-out split")
            confirm = run_holdout_confirm(
                report.to_dict(), holdout_rows,
                outcome_col=outcome_col or "label",
                judge=judge, judge_meta=judge_meta,
                group_col=group_col,
            )
            confirm["split_meta"] = split_meta
            confirm["caveats"] = confirm_caveats
            (out_dir / "confirm_report.json").write_text(
                json.dumps(confirm, indent=1, default=str), encoding="utf-8"
            )
            cadj = confirm.get("adjudication") or {}
            if progress_sink is not None:
                progress_sink.emit(
                    "m4", "completed", "Held-out verification complete",
                    metrics={
                        "n_rows": confirm.get("n_validate_rows"),
                        "n_rejected": cadj.get("n_rejected", 0),
                        "n_adjudicated": cadj.get("n_host_adjudicated", 0),
                    },
                )
            print(
                f"\nheld-out adjudication: {cadj.get('n_rejected', 0)}/"
                f"{cadj.get('n_host_adjudicated', 0)} reject "
                f"(n={confirm.get('n_validate_rows')} rows, "
                f"{confirm.get('n_validate_fail')} failures; correction: "
                f"{cadj.get('method')} over {cadj.get('n_in_family', 0)} claim(s)"
                + (f", groups: {group_col}" if group_col else "") + ")"
            )
            for v in confirm.get("signal_verdicts", []):
                if v.get("status") != "adjudicated":
                    continue
                claim = v.get("claim") or {}
                ci = v.get("ci") or (float("nan"), float("nan"))
                print(f" - {v.get('name')} [{v.get('design')}]: effect "
                      f"{v.get('effect'):+.4f} (CI {ci[0]:+.4f}..{ci[1]:+.4f}), "
                      f"claim \"{claim.get('text')}\" {claim.get('status')}")
            for v in confirm.get("hypothesis_verdicts", []):
                host = v.get("host_verdict")
                tag = f"{v.get('verdict')}" + (f"; host: {host}" if host not in (None, "none") else "")
                print(f" * [{tag}] {str(v.get('statement', ''))[:90]}")
        else:
            (out_dir / "holdout_split.json").write_text(
                json.dumps(split_meta, indent=1), encoding="utf-8"
            )

    # The report is written even when no browser is opened, so every
    # completed exploratory run is portable and reviewable.
    from evalrx.reporting.static_export import write_report

    write_report(out_dir, open_report=open_report)
    return 0 if report.ok else 1


def _group_checks(
    records: "list[dict[str, Any]]",
    explore_rows: "list[dict[str, Any]]",
    holdout_rows: "list[dict[str, Any]]",
    *,
    outcome_col: str | None,
    group_col: str | None,
    time_col: str | None,
) -> "tuple[dict[str, Any], list[str]]":
    """Check the split against the data's grouping; return split facts and warnings.

    With a group column the overlap is zero by construction and is recorded as
    such. Without one, each column that looks like a grouping unit is reported
    with the share of held-out rows whose group the explorer has seen: when
    that share is large the "held-out" rows are near-copies of explored ones.
    Columns that almost determine the outcome are reported as well.
    """
    from evalrx.analysis.fused_pipeline import group_overlap
    from evalrx.analysis.profile import profile_records

    facts: dict[str, Any] = {}
    caveats: list[str] = []
    if group_col:
        if not any(r.get(group_col) not in (None, "") for r in records):
            caveats.append(
                f"group column {group_col!r} is absent or empty in the loaded rows; "
                "the split and the intervals treated rows as independent"
            )
        else:
            facts["group_overlap"] = group_overlap(explore_rows, holdout_rows, group_col)
    profile = profile_records(
        records, outcome_col=outcome_col,
        group_cols=[group_col] if group_col else (),
        time_cols=[time_col] if time_col else (),
    )
    suggested = []
    for s in profile.suggested_group_columns:
        overlap = group_overlap(explore_rows, holdout_rows, s["column"])
        suggested.append({**s, "holdout_share_in_explore_groups": overlap["share"]})
        if group_col:
            continue
        caveats.append(
            f"column {s['column']!r} looks like the unit rows were cut from "
            f"({s['n_groups']} groups; rows of one group share {s['icc']:.0%} of the "
            f"outcome variance) and {overlap['share']:.1%} of held-out rows share "
            "a group with an explored row, so the held-out check is not independent "
            f"of the exploration. Re-run with --group-col {s['column']}."
        )
    if suggested:
        facts["suggested_group_columns"] = suggested
    for s in profile.leakage_suspects:
        caveats.append(
            f"column {s['column']!r} almost fully determines the outcome "
            f"({s['measure']}={s['value']}); a signal built on it may be the "
            "outcome under another name"
        )
    if profile.leakage_suspects:
        facts["leakage_suspects"] = profile.leakage_suspects
    return facts, caveats


def _keyword_candidates(
    explore_rows: "list[dict[str, Any]]", text_col: str, outcome_col: str
) -> list[Any]:
    """Keyword recipes chosen on the exploration rows (see ``analysis.keywords``)."""
    from evalrx.analysis.holdout import outcome_values
    from evalrx.analysis.keywords import keyword_candidates

    values, kind, _, _ = outcome_values(explore_rows, outcome_col)
    if kind is None:
        return []
    if kind == "continuous":
        present = sorted(v for v in values if v is not None)
        median = present[len(present) // 2]
        side: "list[float | None]" = [
            None if v is None else (1.0 if v > median else 0.0) for v in values
        ]
        return keyword_candidates(explore_rows, text_col, side,
                                  side_names=("above-median", "at-or-below-median"))
    return keyword_candidates(explore_rows, text_col, values)


def _verdict_suffix(signal: Any) -> str:
    """One-line host verdict tag for a candidate signal, or '' if not adjudicated."""
    if not getattr(signal, "host_adjudicated", False):
        return "  [descriptive]" if getattr(signal, "sufficient", None) else ""
    verdict = "REJECT H0" if signal.reject else "inconclusive"
    parts = [verdict]
    if signal.e_value is not None:
        # In the e-BH family (has an e-value); reject already reflects survival.
        parts.append(f"e={signal.e_value:.2f}")
        parts.append("e-BH family")
    elif signal.ci is not None:
        parts.append(f"CI={signal.ci[0]:+.3f}..{signal.ci[1]:+.3f}")
        if getattr(signal, "p_value", None) is not None:
            parts.append(f"p={signal.p_value:.3g}")
            parts.append("BH family")
    return "  [host: " + ", ".join(parts) + "]"


def write_report_artifacts(
    report: Any, out_dir: Path, *, report_filename: str = "exploratory_report.json"
) -> None:
    """Persist one explore run's artifacts: figures/, tables/, rendered charts,
    the report JSON, and the generated code/stdout/stderr.

    Charts are rendered here (host-side, from spec + CSV) so the persisted
    ``exploratory_report.json`` carries each chart's ``figure_path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_artifact_dirs(report, out_dir)

    # Render the explorer's chart specs to PNG from the copied tables/ CSVs.
    try:
        report.charts = render_chart_specs(
            getattr(report, "charts", None), out_dir / "tables", out_dir
        )
    except Exception:  # rendering is best-effort, never blocks persistence
        pass

    (out_dir / report_filename).write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    if report.code:
        (out_dir / "analysis.py").write_text(report.code, encoding="utf-8")
    if report.stdout:
        (out_dir / "stdout.txt").write_text(report.stdout, encoding="utf-8")
    if report.stderr:
        (out_dir / "stderr.txt").write_text(report.stderr, encoding="utf-8")
    if report.raw_outputs:
        (out_dir / "agent_raw_output.txt").write_text(
            "\n\n--- attempt ---\n\n".join(report.raw_outputs),
            encoding="utf-8",
        )
    # The UNTRUNCATED raw CLI streams (one per attempt) — the truncated
    # rendering above is for compact UIs; this is the full audit trail.
    raw_streams = getattr(report, "raw_streams", None) or []
    if any(s.strip() for s in raw_streams):
        (out_dir / "agent_raw_streams.txt").write_text(
            "\n\n--- attempt (full stream) ---\n\n".join(raw_streams),
            encoding="utf-8",
        )
    if report.agent_audits:
        (out_dir / "agent_audit.json").write_text(
            json.dumps({"schema_version": 1, "attempts": report.agent_audits}, indent=2),
            encoding="utf-8",
        )

    # Clean up redundant nested sandbox directory so out_dir is the single clean source of truth
    workdir = Path(getattr(report, "workdir", "") or "")
    try:
        if workdir.exists() and workdir.resolve() != out_dir.resolve() and workdir.is_relative_to(out_dir):
            shutil.rmtree(workdir)
    except Exception:
        pass


def _copy_artifact_dirs(report: Any, out_dir: Path) -> None:
    workdir = Path(getattr(report, "workdir", "") or "")
    if not workdir.exists():
        return
    for name in ("figures", "tables"):
        src = workdir / name
        dest = out_dir / name
        if src.exists() and src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
    # Persist the raw loaded records alongside the report — otherwise they only
    # live in the sandbox workdir, which isn't guaranteed to still exist by the
    # time someone reads the report.
    records_src = workdir / RECORDS_FILENAME
    if records_src.exists():
        shutil.copy2(records_src, out_dir / RECORDS_FILENAME)
