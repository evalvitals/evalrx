"""RunContext — single owner of one diagnosis run's output directory.

Historically each ``examples/*/run.py`` glued together four independent output
producers by hand: the example wrote report files at the run root, the
:class:`~evalvitals.eval_agent.run_logger.RunLogger` was buried under a
``logs/`` subdir, :class:`StatsAnalysisAgent` figures went to a hand-built
``logs/figures/`` path, and the sandbox wrote experiment code into an ephemeral
temp dir.  ``RunContext`` replaces that gluing with one library-owned object
that owns the whole run directory and hands every producer its subdirectory.

Layout (single root, no ``logs/`` nesting)::

    <root>/
    ├── manifest.json     run config + index of every produced file
    ├── run_log.jsonl     structured event stream (RunLogger)
    ├── README.txt        auto-generated file guide (from manifest)
    ├── report/           human deliverables (summary.md, hypotheses.json, …)
    ├── figures/          M1 heatmaps + M2 effect plots
    ├── explore/          optional in-cycle explore step: exploratory_report.json,
    │                     tables/*.csv, figures/*.png (VLDiagnoseLoop(explorer=...))
    ├── artifacts/        M1 heavy numeric data (.npy / .json)
    ├── prompts/          judge prompt / response
    ├── experiments/      one self-contained folder per M4 experiment (see new_trial)
    ├── tools/            synthesised probe / stats tool code (M1/M2, run-global)
    ├── workspace/         sandbox working dirs outside any trial
    └── fixes/            one self-contained folder per repair attempt + outcome.md

``fixes/`` and ``experiments/`` are further split into *trials*
(:meth:`RunContext.new_trial`) — one numbered folder per attempt holding its
generated code, the sandbox it ran in, judge prompt/output, and its
record.md + result.json, so "what did attempt #14 do" is one folder, not a
filename-slug hunt across ``tools/`` / ``workspace/`` / ``fixes/``.

Usage::

    from evalvitals.eval_agent import RunContext, VLDiagnoseLoop

    with RunContext("examples/foo/outputs", verbose=True) as ctx:
        stats_agent = StatsAnalysisAgent(judge=judge, figure_dir=str(ctx.figures_dir))
        loop = VLDiagnoseLoop(..., run_logger=ctx.logger)
        report = loop.run(cases)
        ctx.write_diagnose_report(report, cases, discovery=discovery_rows)
    # manifest.json + README.txt written, logger closed on exit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalvitals.eval_agent.run_logger import RunLogger


# Human-readable descriptions for the auto-generated README, keyed by the
# top-level subdirectory name.  Files that do not fall under a known category
# are grouped under "other".
_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "report": "human-facing deliverables: run summary, hypotheses, M5 results",
    "figures": "plots: M1 attention/spatial heatmaps and M2 effect-size charts",
    "explore": "in-cycle explore step (VLDiagnoseLoop(explorer=...)): free-form EDA "
               "beside the catalog M2 — exploratory_report.json, tables/*.csv, "
               "figures/*.png; descriptive notes M3 was shown, never M2/M5 evidence",
    "artifacts": "M1 heavy numeric data (.npy tensors, .json finding dumps)",
    "prompts": "verbatim judge prompt + response for each M1/M2/M3 call",
    "experiments": "one self-contained folder per M4 mechanism-verification "
                   "experiment (code + sandbox + record.md), see new_trial()",
    "tools": "code the agent synthesised for new probes / stats tools (M1/M2)",
    "workspace": "sandbox working directories outside any trial",
    "fixes": "one self-contained folder per repair attempt (code + sandbox + "
             "record.md + result.json), see new_trial(); outcome.md summarises all",
}

# Order categories appear in the README / manifest.
_CATEGORY_ORDER = [
    "report", "figures", "explore", "artifacts", "prompts",
    "experiments", "tools", "workspace", "fixes", "other",
]


class Trial:
    """One self-contained attempt — a fix candidate or an M4 experiment.

    Returned by :meth:`RunContext.new_trial`.  Everything about this one
    attempt (generated code, the sandbox it ran in, judge prompt/output, and
    its result/record) is written under :attr:`root`, so reviewing "what did
    attempt #14 do" never requires hopping across run-global category dirs.

    Directory creation is lazy: nothing is written to disk until the first
    call to :meth:`write` / :attr:`workspace`, so an attempt that is discarded
    before producing anything leaves no empty folder behind.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._workspace: "Path | None" = None

    @property
    def workspace(self) -> Path:
        """``<trial>/workspace/`` — where the sandbox for this attempt runs."""
        if self._workspace is None:
            self._workspace = self.root / "workspace"
            self._workspace.mkdir(parents=True, exist_ok=True)
        return self._workspace

    def write(self, name: str, content: "str | bytes") -> Path:
        """Write *content* to ``<trial>/<name>``; return the path."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def write_record(self, markdown: str) -> Path:
        """Write the human-readable summary to ``<trial>/record.md``."""
        return self.write("record.md", markdown)

    def write_result(self, data: "dict[str, Any]") -> Path:
        """Write the machine-readable outcome to ``<trial>/result.json``."""
        return self.write("result.json", json.dumps(data, indent=2, default=str))

    def __repr__(self) -> str:
        return f"Trial(root={str(self.root)!r})"


class RunContext:
    """Owns one run's output directory and every producer's subdirectory.

    Args:
        root:     Run root directory.  Created if missing.  Defaults to
                  ``runs/<YYYYMMDD_HHMMSS>/`` relative to cwd.
        run_id:   Optional identifier recorded in the manifest; defaults to the
                  root directory name.
        verbose:  Forwarded to the :class:`RunLogger` (human-readable stdout).
        config:   Optional run-configuration dict recorded verbatim in the
                  manifest (model, judge, protocol, …).
    """

    def __init__(
        self,
        root: "str | Path | None" = None,
        *,
        run_id: "str | None" = None,
        verbose: bool = False,
        config: "dict[str, Any] | None" = None,
    ) -> None:
        if root is None:
            root = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        # Resolved to absolute: sandbox subprocesses run with cwd=<some
        # workdir under root> *and* a script path built from that same
        # workdir (see ExperimentSandbox._run_script / run_coded_pipeline) —
        # if root stayed relative, the child process would resolve that
        # script path a second time relative to its new cwd, doubling it.
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.root.name
        self.config = dict(config or {})
        self._verbose = verbose
        self._logger: "RunLogger | None" = None
        self._workdir_seq = 0
        self._trial_seq: "dict[str, int]" = {}

    # ------------------------------------------------------------------
    # Directory properties — each lazily created on first access.
    # ------------------------------------------------------------------

    def _sub(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def report_dir(self) -> Path:
        return self._sub("report")

    @property
    def figures_dir(self) -> Path:
        return self._sub("figures")

    @property
    def explore_dir(self) -> Path:
        """``explore/`` — the in-cycle explore step's report, tables and rendered
        figures (``VLDiagnoseLoop(explorer=..., explore_dir=ctx.explore_dir)``;
        the loop derives the same path from ``ctx.logger`` when not given)."""
        return self._sub("explore")

    @property
    def artifacts_dir(self) -> Path:
        return self._sub("artifacts")

    @property
    def prompts_dir(self) -> Path:
        return self._sub("prompts")

    @property
    def experiments_dir(self) -> Path:
        return self._sub("experiments")

    @property
    def tools_dir(self) -> Path:
        return self._sub("tools")

    @property
    def workspace_dir(self) -> Path:
        return self._sub("workspace")

    @property
    def fixes_dir(self) -> Path:
        return self._sub("fixes")

    @property
    def log_path(self) -> Path:
        return self.root / "run_log.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ------------------------------------------------------------------
    # Logging component
    # ------------------------------------------------------------------

    @property
    def logger(self) -> "RunLogger":
        """The :class:`RunLogger` bound to this context (created on first use)."""
        if self._logger is None:
            from evalvitals.eval_agent.run_logger import RunLogger

            self._logger = RunLogger(context=self, verbose=self._verbose)
        return self._logger

    # ------------------------------------------------------------------
    # Producer-facing path allocation
    # ------------------------------------------------------------------

    def new_workdir(self, label: str) -> Path:
        """Return a fresh, durable sandbox working directory under ``workspace/``.

        Replaces ``tempfile.mkdtemp()`` so the experiment code the agent writes
        is persisted with the rest of the run instead of being deleted.  *label*
        is slugified; a monotonic counter guarantees uniqueness.
        """
        self._workdir_seq += 1
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "work"
        d = self.workspace_dir / f"{self._workdir_seq:02d}_{slug}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def new_trial(self, category: str, label: str) -> "Trial":
        """Allocate one self-contained attempt folder under ``<category>/``.

        A *trial* is one fix candidate or one M4 experiment: its generated
        code, the sandbox it actually ran in, judge prompt/output, and its
        result/record all live together under one numbered folder, instead of
        being scattered across ``tools/`` / ``workspace/`` / ``fixes/`` and
        re-correlated by filename slug. *category* is ``"fixes"`` or
        ``"experiments"``; numbering is monotonic per category.

        The trial's own folder (and its ``workspace/``) is created lazily on
        first write — a candidate discarded before producing anything (e.g. a
        deduped fix proposal) leaves no empty folder on disk; the numbering
        still advances, so a gap in the sequence honestly means "proposed,
        then discarded," not a missing record.
        """
        if category == "fixes":
            parent = self.fixes_dir
        elif category == "experiments":
            parent = self.experiments_dir
        else:
            raise ValueError(
                f"new_trial: unknown category {category!r} "
                "(expected 'fixes' or 'experiments')"
            )
        seq = self._trial_seq.get(category, 0) + 1
        self._trial_seq[category] = seq
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "trial"
        return Trial(parent / f"{seq:02d}_{slug}")

    def figure_path(self, name: str) -> Path:
        """Return ``figures/<name>`` (figures dir created if needed)."""
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            name = f"{name}.png"
        return self.figures_dir / name

    # ------------------------------------------------------------------
    # Report API — absorbs the per-example boilerplate.
    # ------------------------------------------------------------------

    def write_report_file(self, name: str, content: "str | bytes") -> Path:
        """Write *content* to ``report/<name>``; return the path."""
        path = self.report_dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def write_diagnose_report(
        self,
        report: Any,
        cases: "list[Any]",
        *,
        discovery: "list[dict[str, Any]] | None" = None,
    ) -> "dict[str, Path]":
        """Write the standard ``report/`` deliverables from a diagnose report.

        *report* is an :class:`AutoDiagnoseReport` (all three loops return
        this one unified class — ``VLDiagnoseReport`` is an alias for it);
        the ``getattr`` fallbacks below stay duck-typed so a hand-built
        ``SimpleNamespace`` with only a subset of fields (e.g. in tests)
        still writes cleanly. This is the single home for the flattening
        logic previously copy-pasted into every example's
        ``_write_report_artifacts``.

        *discovery* (optional) is a list of already-serialised case rows written
        verbatim to ``report/discovery_cases.json`` — examples that compute
        task-specific columns (e.g. parsed yes/no) build the rows themselves.
        """
        hyps_src = getattr(report, "all_hypotheses", None)
        if hyps_src is None:
            hyps_src = getattr(report, "final_hypotheses", [])
        hypotheses = [
            {
                "statement": h.statement,
                "failure_mode": h.predicted_failure_mode,
                "status": h.status.value if h.status else None,
            }
            for h in hyps_src
        ]
        m5_results = [
            {
                "hypothesis": tr.hypothesis.statement,
                "failure_mode": tr.hypothesis.predicted_failure_mode,
                "status": tr.status.value,
                "effect_size": tr.effect_size,
                "confidence": tr.confidence,
                "protocol_consistent": tr.is_consistent_with_protocol,
                "verdict": tr.verdict,
                "evidence": tr.evidence,
            }
            for tr in getattr(report, "all_test_results", [])
        ]
        n_verified = len(getattr(report, "verified_hypotheses", []))
        summary = {
            "run_id": self.run_id,
            "cycles": report.cycles,
            "stopped_by": getattr(report, "stopped_by", None),
            "resolved": getattr(report, "resolved", None),
            "n_cases": len(cases),
            "n_hypotheses": len(hypotheses),
            "n_verified": n_verified,
        }

        written: dict[str, Path] = {}
        written["hypotheses"] = self.write_report_file(
            "hypotheses.json", json.dumps(hypotheses, indent=2, default=str)
        )
        written["m5_results"] = self.write_report_file(
            "m5_results.json", json.dumps(m5_results, indent=2, default=str)
        )
        written["summary_json"] = self.write_report_file(
            "summary.json", json.dumps(summary, indent=2, default=str)
        )

        lines = [
            f"# {self.run_id} — Run Summary",
            "",
            f"- stopped_by: {summary['stopped_by']}",
            f"- resolved: {summary['resolved']}",
            f"- cycles: {summary['cycles']}",
            f"- cases: {summary['n_cases']}",
            f"- hypotheses: {summary['n_hypotheses']}",
            f"- verified: {summary['n_verified']}",
            "",
            "## Hypotheses",
        ]
        for h in hypotheses or [
            {"status": None, "failure_mode": "none", "statement": "none"}
        ]:
            lines.append(f"- [{h['status']}] {h['failure_mode']}: {h['statement']}")
        written["summary_md"] = self.write_report_file(
            "summary.md", "\n".join(lines) + "\n"
        )

        if discovery is not None:
            written["discovery"] = self.write_report_file(
                "discovery_cases.json", json.dumps(discovery, indent=2, default=str)
            )
        return written

    # ------------------------------------------------------------------
    # Manifest + README — built by walking the tree at finalize().
    # ------------------------------------------------------------------

    def _scan(self) -> "dict[str, list[str]]":
        """Group every file under root by its top-level category subdirectory."""
        by_cat: dict[str, list[str]] = {}
        for f in sorted(self.root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(self.root)
            if rel.name in ("manifest.json", "README.txt"):
                continue
            top = rel.parts[0] if len(rel.parts) > 1 else "other"
            if top not in _CATEGORY_DESCRIPTIONS and top != "run_log.jsonl":
                top = "other" if len(rel.parts) == 1 else top
            cat = top if top in _CATEGORY_DESCRIPTIONS else "other"
            by_cat.setdefault(cat, []).append(str(rel))
        return by_cat

    def write_manifest(self) -> Path:
        """Write ``manifest.json`` — run config + categorised file registry."""
        by_cat = self._scan()
        manifest = {
            "run_id": self.run_id,
            "root": str(self.root),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.config,
            "files": {
                cat: by_cat[cat]
                for cat in _CATEGORY_ORDER
                if by_cat.get(cat)
            },
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return self.manifest_path

    def write_readme(self) -> Path:
        """Generate ``README.txt`` from the on-disk layout (no hardcoded paths)."""
        by_cat = self._scan()
        lines = [f"{self.root.name}/  —  run output guide", ""]
        if (self.root / "run_log.jsonl").exists():
            lines += ["run_log.jsonl", "    one JSON line per M1/M2/M3/M4/M5 event", ""]
        for cat in _CATEGORY_ORDER:
            files = by_cat.get(cat)
            if not files:
                continue
            lines.append(f"{cat}/")
            lines.append(f"    {_CATEGORY_DESCRIPTIONS.get(cat, 'misc files')}")
            for rel in files[:12]:
                lines.append(f"      {rel}")
            if len(files) > 12:
                lines.append(f"      … (+{len(files) - 12} more)")
            lines.append("")
        path = self.root / "README.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Write the manifest + README and close the logger.  Idempotent."""
        if self._logger is not None:
            self._logger.close()
        self.write_manifest()
        self.write_readme()

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finalize()

    def __repr__(self) -> str:
        return f"RunContext(root={str(self.root)!r})"
