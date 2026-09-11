"""RunContext — single owner of one diagnosis run's output directory.

Historically each ``examples/*/run.py`` glued together four independent output
producers by hand: the example wrote report files at the run root, the run
logger was buried under a ``logs/`` subdir, :class:`StatsAnalysisAgent`
figures went to a hand-built ``logs/figures/`` path, and the sandbox wrote
experiment code into an ephemeral temp dir.  ``RunContext`` replaces that
gluing with one library-owned object that owns the whole run directory and
hands every producer its subdirectory.

The logger is :class:`~evalrx.eval_agent.run_logger_v2.RunLoggerV2` (see
``evalrx/eval_agent/RUN_LOGGER_V2.md``): one ``run.json`` plus one
``M1/log.json``..``M5/log.json`` per stage, no flat event file, no persisted
``prompts/``/``experiments/``/``tools/``/``workspace/``/``fixes/`` — generated
text/code is captured inline into the relevant stage's JSON via an ephemeral
runtime tree that :meth:`finalize` deletes, and no ``manifest.json``/
``README.txt`` is written (the manifest lives inside ``run.json`` instead).

Layout::

    <root>/
    ├── run.json           run-wide events: run_start, cases, diagnose_reports, manifest, …
    ├── M1/log.json …      one JSON document per stage, plus each stage's own artifacts/
    │   M5/log.json
    ├── contract/          one validated JSON per stage (see evalrx.contract), if emitted
    ├── artifacts/         M1 heavy numeric data (.npy / .json) written outside any stage
    └── langfuse_trace.json

``fixes/`` and ``experiments/`` attempts (:meth:`RunContext.new_trial`) — one
numbered attempt per fix candidate or M5 experiment — live under an ephemeral
:attr:`runtime_root` instead: generated code, the sandbox it ran in, and
judge prompt/output are inlined into the owning stage's JSON, and the
runtime tree is deleted at :meth:`finalize`, so "what did attempt #14 do" is
one entry in that stage's log, not a filename-slug hunt across loose folders.

Usage::

    from evalrx.eval_agent import RunContext, VLDiagnoseLoop

    with RunContext("examples/foo/outputs", verbose=True) as ctx:
        stats_agent = StatsAnalysisAgent(judge=judge, figure_dir=str(ctx.figures_dir))
        loop = VLDiagnoseLoop(..., run_logger=ctx.logger)
        report = loop.run(cases)
        ctx.write_diagnose_report(report, cases, discovery=discovery_rows)
    # run.json/M*/log.json are flushed incrementally throughout the run;
    # finalize() just closes them out and inlines the runtime tree.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2


class Trial:
    """One self-contained attempt — a fix candidate or an M5 experiment.

    Returned by :meth:`RunContext.new_trial`.  Everything about this one
    attempt (generated code, the sandbox it ran in, judge prompt/output, and
    its result/record) is written under :attr:`root`, so reviewing "what did
    attempt #14 do" never requires hopping across run-global category dirs.
    :attr:`root` lives under :attr:`RunContext.runtime_root` — an ephemeral
    tree captured into the owning stage's JSON and deleted at
    :meth:`RunContext.finalize`, not a permanent directory.

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
        verbose:  Forwarded to the logger (human-readable stdout).
        narrate:  Forwarded to the logger — live, aligned M1-M5 terminal
                  narration instead of `verbose`'s raw one-liner (see
                  :class:`evalrx.eval_agent.narration.LoopNarrator`).
                  Supersedes *verbose* rather than stacking with it.
        config:   Optional run-configuration dict recorded verbatim in the
                  manifest (model, judge, protocol, …).

    The logger allocates producer sandboxes in an external ephemeral runtime
    tree; their text is inlined into JSON and their media is copied before
    finalization removes the tree.
    """

    def __init__(
        self,
        root: "str | Path | None" = None,
        *,
        run_id: "str | None" = None,
        verbose: bool = False,
        narrate: bool = False,
        config: "dict[str, Any] | None" = None,
        observability_mode: str | None = None,
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
        self._narrate = narrate
        self._observability_mode = observability_mode
        self._logger: "RunLoggerV2 | None" = None
        self._workdir_seq = 0
        self._trial_seq: "dict[str, int]" = {}
        self._runtime_root: "Path | None" = None
        self._finalized = False

    @property
    def runtime_root(self) -> Path:
        """Ephemeral execution workspace used by producers.

        Generated text/code is captured by the logger before this tree is
        removed; it never becomes part of the portable run artifact.
        """
        if self._runtime_root is None:
            self._runtime_root = Path(tempfile.mkdtemp(prefix=f"evalrx-{self.run_id}-"))
        return self._runtime_root

    # ------------------------------------------------------------------
    # Directory properties — each lazily created on first access.
    # ------------------------------------------------------------------

    def _sub(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figures_dir(self) -> Path:
        return self._sub("M2/artifacts")

    @property
    def explore_dir(self) -> Path:
        """The in-cycle explore step's report, tables and rendered figures
        (``VLDiagnoseLoop(explorer=..., explore_dir=ctx.explore_dir)``; the
        loop derives the same path from ``ctx.logger`` when not given).
        Ephemeral — captured into ``M2/log.json`` and deleted at
        :meth:`finalize`, not a permanent directory."""
        d = self.runtime_root / "explore"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def artifacts_dir(self) -> Path:
        return self._sub("artifacts")

    # ------------------------------------------------------------------
    # Logging component
    # ------------------------------------------------------------------

    @property
    def logger(self) -> "RunLoggerV2":
        """The logger bound to this context (created on first use)."""
        if self._logger is None:
            from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

            self._logger = RunLoggerV2(
                run_dir=self.root, verbose=self._verbose, narrate=self._narrate,
                observability_mode=self._observability_mode,
                context=self,
            )
        return self._logger

    # ------------------------------------------------------------------
    # Producer-facing path allocation
    # ------------------------------------------------------------------

    def new_workdir(self, label: str) -> Path:
        """Return a fresh sandbox working directory under the runtime tree.

        Replaces ``tempfile.mkdtemp()`` so the experiment code the agent
        writes is captured with the rest of the run instead of vanishing
        untracked. *label* is slugified; a monotonic counter guarantees
        uniqueness.
        """
        self._workdir_seq += 1
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "work"
        d = self.runtime_root / "workspace" / f"{self._workdir_seq:02d}_{slug}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def new_trial(self, category: str, label: str) -> "Trial":
        """Allocate one self-contained attempt folder under the runtime tree.

        A *trial* is one fix candidate or one M5 experiment: its generated
        code, the sandbox it actually ran in, judge prompt/output, and its
        result/record all live together under one numbered folder.
        *category* is ``"fixes"`` or ``"experiments"``; numbering is
        monotonic per category.

        The trial's own folder (and its ``workspace/``) is created lazily on
        first write — a candidate discarded before producing anything (e.g. a
        deduped fix proposal) leaves no empty folder on disk; the numbering
        still advances, so a gap in the sequence honestly means "proposed,
        then discarded," not a missing record.
        """
        if category not in ("fixes", "experiments"):
            raise ValueError(
                f"new_trial: unknown category {category!r} "
                "(expected 'fixes' or 'experiments')"
            )
        seq = self._trial_seq.get(category, 0) + 1
        self._trial_seq[category] = seq
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_") or "trial"
        return Trial(self.runtime_root / category / f"{seq:02d}_{slug}")

    def figure_path(self, name: str) -> Path:
        """Return ``figures_dir/<name>`` (created if needed)."""
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            name = f"{name}.png"
        return self.figures_dir / name

    # ------------------------------------------------------------------
    # Report API — absorbs the per-example boilerplate.
    # ------------------------------------------------------------------

    def write_diagnose_report(
        self,
        report: Any,
        cases: "list[Any]",
        *,
        discovery: "list[dict[str, Any]] | None" = None,
    ) -> None:
        """Inline the standard post-diagnosis report into ``run.json``.

        *report* is an :class:`AutoDiagnoseReport` (all three loops return
        this one unified class — ``VLDiagnoseReport`` is an alias for it).
        *discovery* (optional) is a list of already-serialised case rows —
        examples that compute task-specific columns (e.g. parsed yes/no)
        build the rows themselves. Delegates to
        :meth:`~evalrx.eval_agent.run_logger_v2.RunLoggerV2.log_diagnose_report`.
        """
        self.logger.log_diagnose_report(report, cases, discovery=discovery)

    def publish_report(
        self,
        *,
        model: "Any | None" = None,
        example_dir: "str | Path | None" = None,
    ) -> "Any":
        """Publish the dynamic report after all diagnostic/fix stages finish."""
        from evalrx.reporting.dynamic import publish_report

        return publish_report(
            self.root, example_dir=example_dir, model=model, run_logger=self._logger,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Close the logger, inline the runtime tree, and delete it.  Idempotent."""
        if self._finalized:
            return
        if self._logger is not None or self._runtime_root is not None:
            logger = self.logger
            if self._runtime_root is not None:
                logger.log_runtime_snapshot(self._runtime_root)
            logger.close()
            # close() materializes the trace bundle and every M1-M5 log;
            # index only afterwards so the manifest is complete.
            logger.log_manifest(run_id=self.run_id, config=self.config)
            # write_contract_index() only looks for an on-disk contract/ dir
            # and the logger's trace_id. Without it, a run that emitted
            # contract/ payloads (evalrx.contract.emit, independent of
            # RunContext) never gets the index.json a reader opening the run
            # without its producer needs.
            self.write_contract_index()
        if self._runtime_root is not None:
            shutil.rmtree(self._runtime_root, ignore_errors=True)
        self._finalized = True

    def write_contract_index(self) -> "Path | None":
        """Write ``contract/index.json`` when any stage emitted a payload.

        Written here rather than by the emitter so it describes the finished
        directory: a reader opening this run — or a zip of it — gets one file
        naming every payload, its stage and cycle, and the wire model that
        decodes it, without having to infer any of that from filenames.
        """
        if not (self.root / "contract").is_dir():
            return None
        try:
            from evalrx.contract.emit import write_index
        except ImportError:  # contract extra not installed
            return None
        trace_id = getattr(self._logger, "trace_id", "") if self._logger else ""
        return write_index(self.root, trace_id=trace_id)

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finalize()

    def __repr__(self) -> str:
        return f"RunContext(root={str(self.root)!r})"
