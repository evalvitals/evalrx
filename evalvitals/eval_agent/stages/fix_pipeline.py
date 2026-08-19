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
  helper writes a ``@@MODEL_CALL@@{json}`` line to stdout and waits for the
  reply on stdin; the host services each call (applies catalog image tools to
  the case's image, runs ``model.generate``) under a per-session call budget
  and wall-clock deadline.  Requests carry a ``rid`` echoed by the reply, so
  the sandbox helper is thread-safe and the host can service several calls at
  once (``concurrency=``) — a pipeline that fans out over cases with a thread
  pool really runs in parallel.
* The script's final ``FIX_PIPELINE_RESULT_JSON=`` line carries per-case final
  answers; **scoring stays host-side** — the case payload shipped to the
  sandbox contains only ``id``, ``prompt`` and the model's own
  ``baseline_output`` (no labels, no expected rubric), so generated code
  cannot cheat by echoing gold answers or flipping known failures.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
import threading as _threading
import itertools as _itertools

# Bridge plumbing — request-id tagged so that CONCURRENT callers (threads) each
# get their own reply.  One reader thread owns stdin and routes replies by id.
_bridge_lock = _threading.Lock()
_bridge_pending = {{}}
_bridge_ids = _itertools.count(1)
_bridge_reader = []


def _bridge_read_loop():
    while True:
        line = _sys.stdin.readline()
        if not line:
            with _bridge_lock:
                for slot in _bridge_pending.values():
                    slot[1] = {{"error": "model bridge closed"}}
                    slot[0].set()
            return
        try:
            resp = _json.loads(line)
        except Exception:
            continue
        with _bridge_lock:
            slot = _bridge_pending.get(resp.get("rid"))
        if slot is not None:
            slot[1] = resp
            slot[0].set()


def _bridge_call(payload):
    rid = next(_bridge_ids)
    done = _threading.Event()
    payload["rid"] = rid
    with _bridge_lock:
        if not _bridge_reader:
            t = _threading.Thread(target=_bridge_read_loop, daemon=True)
            t.start()
            _bridge_reader.append(t)
        _bridge_pending[rid] = [done, None]
        _sys.stdout.write("{call_marker}" + _json.dumps(payload) + "\\n")
        _sys.stdout.flush()
    done.wait()
    with _bridge_lock:
        resp = _bridge_pending.pop(rid)[1] or {{"error": "model bridge closed"}}
    if resp.get("error"):
        raise RuntimeError(resp["error"])
    return resp


def model_generate(case_id, prompt=None, image_ops=None, generation_kwargs=None):
    """Call the original model on one case (host-mediated bridge).

    Thread-safe: concurrent calls (e.g. from a ThreadPoolExecutor over cases)
    are serviced in parallel by the host.

    generation_kwargs: optional bounded decoding controls
    ({{"max_tokens", "temperature", "top_p", "stop"}}); anything else is
    dropped host-side, and max_tokens can only RAISE the baseline budget.
    """
    return _bridge_call(
        {{"case_id": case_id, "prompt": prompt, "image_ops": image_ops or [],
          "generation_kwargs": generation_kwargs or {{}}}}).get("output", "")


def model_attend(case_id, prompt=None):
    """Read the model's attention heatmap over image patches (host-mediated).

    Returns {{"grid": [[float, ...], ...], "shape": [H, W]}} — only available
    when the fix tier allows internals read (L3a+) on a white-box model.
    """
    return _bridge_call({{"op": "attend", "case_id": case_id, "prompt": prompt}})

'''

def cases_payload(cases: "CaseBatch") -> "dict[str, Any]":
    """Serialise cases for the sandbox: id, prompt and the model's ORIGINAL
    answer — never a label, expected answer, or rubric.

    ``baseline_output`` is the recorded baseline generation (``case.observed``,
    ``None`` when the case carries none). It is not a label: it is what the
    unchanged model already said, which a scaffold may legitimately compare
    against, vote with, or ask the model to double-check. It carries no
    information about correctness — the frozen-model control replays exactly
    this text, so a pipeline that merely echoes it scores as the baseline.
    """
    out = []
    for c in cases:
        observed = getattr(c, "observed", None)
        out.append({
            "id": c.id,
            "prompt": str(getattr(getattr(c, "inputs", None), "prompt", "")),
            "baseline_output": None if observed is None else str(observed),
        })
    return {"cases": out}


@dataclass
class CodedPipelineResult:
    """Outcome of one bridged pipeline session."""

    outputs: "dict[str, str]" = field(default_factory=dict)  # case id -> final answer
    n_calls: int = 0
    ok: bool = False
    error: str = ""


def run_coded_pipeline(
    code: str,
    model: "Model | None",
    cases: "CaseBatch",
    workdir: "Path | str",
    timeout_sec: int = 600,
    max_calls: "int | None" = None,
    enable_attend: bool = False,
    reply_fn: "Callable[[Any, str], str] | None" = None,
    max_tokens_floor: "int | None" = None,
    concurrency: int = 1,
) -> CodedPipelineResult:
    """Execute agent-written pipeline *code* with bridged model access.

    The subprocess gets ``fix_cases.json`` + the ``model_generate`` prelude;
    every bridge call is serviced here (image tools applied host-side, model
    invoked host-side).  Returns the per-case final answers for host-side
    scoring.

    ``concurrency``: how many bridged calls the host services at once. Every
    request carries a ``rid`` and the reply echoes it, so the sandbox side is
    thread-safe — generated code that fans out over cases with a thread pool
    really runs in parallel (with ``concurrency=1`` calls are serviced strictly
    in arrival order, the old behaviour). Without the ids a threaded pipeline
    silently read each other's replies: the reply to case A's prompt could
    land in case B's vote.

    ``reply_fn(case, prompt) -> str``, when given, answers every bridged call
    instead of the model (which is then never touched) — the hook
    :func:`frozen_model_control` uses to re-run the same code with the model
    held at its recorded answers.

    ``max_tokens_floor``: the baseline decode budget. A bridged call's
    ``generation_kwargs["max_tokens"]`` below it is raised to it — a scaffold
    may give the model MORE room than the baseline had, never less (a shorter
    budget truncates the chain and scores as a wrong answer, which is a
    decoding artefact, not evidence about the repair).
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
    stdin_lock = threading.Lock()
    pool = (ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
            if int(concurrency) > 1 else None)

    def _rid_of(raw: str) -> "Any":
        try:
            return json.loads(raw).get("rid")
        except Exception:
            return None

    def _send(reply: "dict[str, Any]") -> bool:
        with stdin_lock:
            try:
                proc.stdin.write(json.dumps(reply) + "\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
                return True
            except (BrokenPipeError, OSError, ValueError):
                return False

    def _serve(raw: str) -> None:
        reply = _service_call(raw, case_by_id, model, Inputs, enable_attend,
                              reply_fn, max_tokens_floor=max_tokens_floor)
        reply["rid"] = _rid_of(raw)
        _send(reply)

    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            stripped = line.strip()
            if stripped.startswith(CALL_MARKER):
                res.n_calls += 1
                if res.n_calls > budget or time.monotonic() > deadline:
                    res.error = f"model-call budget exhausted ({budget} calls)"
                    proc.kill()
                    break
                raw = stripped[len(CALL_MARKER):]
                if pool is not None:
                    pool.submit(_serve, raw)
                else:
                    reply = _service_call(raw, case_by_id, model, Inputs,
                                          enable_attend, reply_fn,
                                          max_tokens_floor=max_tokens_floor)
                    reply["rid"] = _rid_of(raw)
                    if not _send(reply):
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
        if pool is not None:
            # In-flight model calls finish on their own (the reply is dropped
            # on a closed pipe); queued ones are cancelled.
            pool.shutdown(wait=False, cancel_futures=True)

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
    model: "Model | None",
    inputs_cls: "type",
    enable_attend: bool = False,
    reply_fn: "Callable[[Any, str], str] | None" = None,
    max_tokens_floor: "int | None" = None,
) -> "dict[str, Any]":
    """Handle one bridged model call; never raises (errors travel as JSON)."""
    import dataclasses

    from evalvitals.eval_agent.stages.fix_tools import _safe_generation_kwargs

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
        if reply_fn is not None:
            # Same contract as the real bridge (unknown case, bad image_ops
            # still error) — only the answer comes from elsewhere.
            return {"output": str(reply_fn(case, str(prompt)))}
        if ops:
            image = apply_image_ops(image, ops)
        if model is None:
            return {"error": "no model behind the bridge"}
        gen_kwargs = _safe_generation_kwargs(req.get("generation_kwargs"))
        if (
            max_tokens_floor
            and "max_tokens" in gen_kwargs
            and gen_kwargs["max_tokens"] < int(max_tokens_floor)
        ):
            gen_kwargs["max_tokens"] = int(max_tokens_floor)
        # dataclasses.replace keeps .video/.audio (a bare Inputs(prompt=,
        # image=) dropped them — the same modality bug run_pipeline had).
        if inp is not None and dataclasses.is_dataclass(inp):
            new_inputs = dataclasses.replace(inp, prompt=str(prompt), image=image)
        else:
            new_inputs = inputs_cls(prompt=str(prompt), image=image)
        return {"output": str(model.generate(new_inputs, **gen_kwargs))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def frozen_model_control(
    code: str,
    cases: "CaseBatch",
    workdir: "Path | str",
    timeout_sec: int = 600,
    max_calls: "int | None" = None,
) -> CodedPipelineResult:
    """Re-run *code* with the model frozen: every bridged call is answered with
    the case's recorded baseline output (``case.observed``; ``""`` when a case
    has none).

    A coded pipeline is only ever asked to repair the MODEL, but nothing in the
    sandbox stops it from repairing the TASK: qwen3.5-2b / bbh_word_sorting
    validated a pipeline whose final answer was ``sorted(input_words(prompt))``
    with the model call accepted only when it already equalled that — 124/125
    correct, e=8.6e15, "FIXED", and 113 model calls that changed nothing. Any
    task with a mechanical oracle (sorting, arithmetic, dates: much of BBH)
    invites the same move, and the timeout-repair round makes it worse.

    Under this control the model cannot behave differently than it already
    did, so a FAILING case that comes out right was solved by the pipeline's
    own computation. The caller excludes such cases from the paired test
    (baseline-correct cases are kept: replaying a right answer proves
    nothing, and dropping them would hide what the candidate breaks).

    Replaying the recorded answer rather than returning ``""`` matters:
    real answers are non-empty and well-formed, so a solver gated on "the
    model said something parseable" is still caught, and a legitimate
    pipeline's benign empty-answer default is not tripped by an artefact of
    the control.
    """
    def _replay(case: "Any", prompt: str) -> str:
        observed = getattr(case, "observed", None)
        return "" if observed is None else str(observed)

    return run_coded_pipeline(
        code, None, cases, workdir=workdir, timeout_sec=timeout_sec,
        max_calls=max_calls, reply_fn=_replay,
    )


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
