"""L2 coded pipelines — agent-written repair scaffolds with bridged model access.

The declarative :class:`~.fix_tools.PipelineSpec` path covers fixed-shape
scaffolds (catalog tools + one templated call).  Real L2 means the coding
agent writes a **brand-new pipeline**: multiple model calls per case, branching
on intermediate outputs (ask for a region → zoom → re-ask, describe → decide,
majority vote, …) — with the single constraint that the *unchanged* model is
used.  Such adaptive pipelines cannot be pre-computed, so the usual
collect-then-compute split does not apply.  Instead:

* The generated code runs in a **sandbox subprocess** (never sees the repo,
  the weights, or the scoring rubric).
* Model access goes through a **bridge**: the injected ``model_generate()``
  helper writes a ``@@MODEL_CALL@@{json}`` line to stdout and reads the reply
  from stdin; the host services each call (applies catalog image tools to the
  case's image, runs ``model.generate``) under a per-session call budget and
  wall-clock deadline.
* The script's final ``FIX_PIPELINE_RESULT_JSON=`` line carries per-case final
  answers; **scoring stays host-side** — the case payload shipped to the
  sandbox contains only ``id`` and ``prompt`` (no labels, no expected rubric),
  so generated code cannot cheat by echoing gold answers or flipping known
  failures.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.eval_agent.stages.fix_tools import apply_image_ops, score_to_bool

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)

CASES_FILENAME = "fix_cases.json"
CALL_MARKER = "@@MODEL_CALL@@"
RESULT_MARKER = "FIX_PIPELINE_RESULT_JSON="

# Injected ahead of the generated code: the only channel to the model.
_PRELUDE = '''\
import json as _json
import sys as _sys


def model_generate(case_id, prompt=None, image_ops=None):
    """Call the original model on one case (host-mediated bridge)."""
    _sys.stdout.write("{call_marker}" + _json.dumps(
        {{"case_id": case_id, "prompt": prompt, "image_ops": image_ops or []}}) + "\\n")
    _sys.stdout.flush()
    line = _sys.stdin.readline()
    if not line:
        raise RuntimeError("model bridge closed")
    resp = _json.loads(line)
    if resp.get("error"):
        raise RuntimeError(resp["error"])
    return resp.get("output", "")


def model_attend(case_id, prompt=None):
    """Read the model's attention heatmap over image patches (host-mediated).

    Returns {{"grid": [[float, ...], ...], "shape": [H, W]}} — only available
    when the fix tier allows internals read (L3a+) on a white-box model.
    """
    _sys.stdout.write("{call_marker}" + _json.dumps(
        {{"op": "attend", "case_id": case_id, "prompt": prompt}}) + "\\n")
    _sys.stdout.flush()
    line = _sys.stdin.readline()
    if not line:
        raise RuntimeError("model bridge closed")
    resp = _json.loads(line)
    if resp.get("error"):
        raise RuntimeError(resp["error"])
    return resp

'''


def cases_payload(cases: "CaseBatch") -> "dict[str, Any]":
    """Serialise cases for the sandbox: id + prompt ONLY (no label/rubric)."""
    return {"cases": [
        {"id": c.id, "prompt": str(getattr(getattr(c, "inputs", None), "prompt", ""))}
        for c in cases
    ]}


@dataclass
class CodedPipelineResult:
    """Outcome of one bridged pipeline session."""

    outputs: "dict[str, str]" = field(default_factory=dict)  # case id -> final answer
    n_calls: int = 0
    ok: bool = False
    error: str = ""


def run_coded_pipeline(
    code: str,
    model: "Model",
    cases: "CaseBatch",
    workdir: "Path | str",
    timeout_sec: int = 600,
    max_calls: "int | None" = None,
    enable_attend: bool = False,
) -> CodedPipelineResult:
    """Execute agent-written pipeline *code* with bridged model access.

    The subprocess gets ``fix_cases.json`` + the ``model_generate`` prelude;
    every bridge call is serviced here (image tools applied host-side, model
    invoked host-side).  Returns the per-case final answers for host-side
    scoring.
    """
    from evalvitals.core.case import Inputs

    res = CodedPipelineResult()
    case_by_id = {c.id: c for c in cases}
    budget = max_calls if max_calls is not None else 6 * len(case_by_id) + 10

    # Resolved to absolute: the subprocess below runs with cwd=workdir *and*
    # a script path built from that same workdir — a relative workdir makes
    # the child resolve the script path a second time relative to its new
    # cwd, doubling it (FileNotFoundError instead of running the script).
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / CASES_FILENAME).write_text(
        json.dumps(cases_payload(cases)), encoding="utf-8")
    script = workdir / "fix_pipeline_exec.py"
    script.write_text(
        _PRELUDE.format(call_marker=CALL_MARKER) + "\n" + code, encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(workdir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stderr_tail: "list[str]" = []

    def _drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line)
            del stderr_tail[:-30]

    threading.Thread(target=_drain_stderr, daemon=True).start()
    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(timeout_sec, _kill_on_timeout)
    watchdog.start()
    deadline = time.monotonic() + timeout_sec

    result_line: "str | None" = None
    unmarked_tail: "list[str]" = []  # fallback: last non-marker stdout lines
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            stripped = line.strip()
            if stripped.startswith(CALL_MARKER):
                res.n_calls += 1
                if res.n_calls > budget or time.monotonic() > deadline:
                    res.error = f"model-call budget exhausted ({budget} calls)"
                    proc.kill()
                    break
                reply = _service_call(stripped[len(CALL_MARKER):], case_by_id,
                                      model, Inputs, enable_attend)
                try:
                    proc.stdin.write(json.dumps(reply) + "\n")  # type: ignore[union-attr]
                    proc.stdin.flush()  # type: ignore[union-attr]
                except (BrokenPipeError, OSError):
                    break
            elif stripped.startswith(RESULT_MARKER):
                result_line = stripped[len(RESULT_MARKER):]
            elif stripped:
                unmarked_tail.append(stripped)
                del unmarked_tail[:-20]
        proc.wait(timeout=10)
    except Exception as exc:
        res.error = res.error or f"bridge session failed: {exc}"
        proc.kill()
    finally:
        watchdog.cancel()

    if result_line is None and not timed_out.is_set():
        # A judge-written pipeline sometimes gets the JSON payload right but
        # drops the exact literal marker prefix the prompt asked for (a
        # compliance slip, not a content error, and the one-shot repair
        # round tends to repeat it). Recover the payload straight from the
        # last unmarked stdout line if it parses as the expected shape,
        # rather than discarding a working pipeline over a missing prefix.
        for candidate_line in reversed(unmarked_tail):
            try:
                parsed = json.loads(candidate_line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("per_case"), list):
                result_line = candidate_line
                logger.info(
                    "fix_pipeline: recovered result JSON without the literal "
                    "%s prefix", RESULT_MARKER,
                )
                break

    if result_line is None:
        if timed_out.is_set():
            # A kill looks like a silent exit (empty stderr) — name the cause,
            # or the caller misreads an under-budgeted run as broken code.
            res.error = res.error or (
                f"timed out after {timeout_sec}s and was killed (bridge served "
                f"{res.n_calls} model calls) — raise exec_timeout_sec or shrink "
                "the validation batch"
            )
        else:
            res.error = res.error or (
                "no FIX_PIPELINE_RESULT_JSON line; stderr tail: "
                + "".join(stderr_tail)[-400:].strip()
            )
        return res
    try:
        per_case = json.loads(result_line).get("per_case", [])
    except json.JSONDecodeError as exc:
        res.error = f"unparseable result line: {exc}"
        return res
    for entry in per_case:
        if isinstance(entry, dict) and str(entry.get("sample_id", "")) in case_by_id:
            res.outputs[str(entry["sample_id"])] = str(entry.get("output", ""))
    res.ok = bool(res.outputs)
    return res


def _service_call(
    raw: str,
    case_by_id: "dict[str, Any]",
    model: "Model",
    inputs_cls: "type",
    enable_attend: bool = False,
) -> "dict[str, Any]":
    """Handle one bridged model call; never raises (errors travel as JSON)."""
    try:
        req = json.loads(raw)
        case = case_by_id.get(str(req.get("case_id", "")))
        if case is None:
            return {"error": f"unknown case_id {req.get('case_id')!r}"}
        if req.get("op") == "attend":
            if not enable_attend:
                return {"error": "model_attend requires fix tier >= L3a on a "
                                 "white-box model"}
            from evalvitals.analyzers.attention.relative_attn import attention_heatmap

            grid = attention_heatmap(model, case)
            if grid is None:
                return {"error": "attention capture failed for this case"}
            return {"grid": grid.tolist(), "shape": list(grid.shape)}
        inp = getattr(case, "inputs", None)
        prompt = req.get("prompt") or str(getattr(inp, "prompt", ""))
        image = getattr(inp, "image", None)
        ops = req.get("image_ops") or []
        if ops:
            # Strict contract: a malformed op silently skipped hides the bug
            # from the coding agent forever; an error reply becomes a
            # RuntimeError inside the generated code — visible and repairable.
            from evalvitals.eval_agent.stages.fix_tools import IMAGE_TOOLS

            bad = [op for op in ops if not isinstance(op, dict) or not op.get("tool")]
            unknown = [str(op["tool"]) for op in ops
                       if isinstance(op, dict) and op.get("tool")
                       and str(op["tool"]) not in IMAGE_TOOLS]
            if bad or unknown:
                return {"error": (
                    "invalid image_ops — each op must be "
                    "{'tool': <name>, 'params': {...}}"
                    + (f"; unknown tool(s): {', '.join(unknown)}" if unknown else "")
                    + "; available tools: " + ", ".join(IMAGE_TOOLS)
                )}
            image = apply_image_ops(image, ops)
        return {"output": str(model.generate(inputs_cls(prompt=str(prompt), image=image)))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def score_outputs(
    result: CodedPipelineResult,
    cases: "CaseBatch",
    score_fn: "Callable[[Any, str], Optional[bool]]",
) -> "dict[str, Optional[bool]]":
    """Host-side scoring of the pipeline's final answers (rubrics never left)."""
    scores: "dict[str, Optional[bool]]" = {}
    for case in cases:
        output = result.outputs.get(case.id)
        scores[case.id] = None if output is None else score_to_bool(score_fn(case, output))
    return scores
