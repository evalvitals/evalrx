"""M1 -> M2 -> M3 -> M5 -> M4 over a frozen CaseBatch from build_cases.py.

    python run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law
    python run_pipeline.py --analysis-only        # M1->M2->M3, stop before M5/M4
    python -m evalvitals.cli dashboard outputs/qwen3.5-9b/supergpqa_law

Stages (evalvitals.eval_agent.loop.VLDiagnoseLoop -- the class name says VL, but
it takes a plain Model plus an ExperimentProtocol whose target_modalities is
{"text"} here, and nothing in it is vision-specific):

    M1 ProbeAgent          selects and runs analyzers against the batch
       ExploratoryAnalysisAgent  (optional, config `explore`) free-form EDA over
                          M1's per-case table: tables/ + rendered figures/ under
                          outputs/<model>/<dataset>/explore/, and UNCONFIRMED
                          notes for M3. Runs beside the catalog M2, not instead.
    M2 StatsAnalysisAgent  protocol-aware statistics over M1's per-case signals
                          (the confirmatory tool catalog + e-BH; unchanged)
    M3 DiagnosisAgent      proposes hypotheses from the stats (+ explore notes)
    M5 HypothesisTester    tests each hypothesis + checks protocol consistency
    M4 SurgeryAgent        proposes a fix for the best VERIFIED hypothesis

M4 runs OUTSIDE the loop, on the held-out confirm split (config `confirm_split`),
so the repair is validated on cases the loop never mined for its hypothesis. With
confirm_split at 0 the fix is scored on the same data that produced the
hypothesis, which is how a diagnosis loop flatters itself.

The judge/coder are `claude -p` CLI calls and are NOT the model under test. The
model under test is only loaded for generation: already done in build_cases.py,
and again by M4 if the fix needs to be executed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))
#: The checkout root (holds pyproject.toml). Found by walking up rather than
#: counting directories: this example moved from examples/ to
#: examples/dataset_selection/ in one reorg, and a hardcoded ``parent.parent``
#: silently started pointing at examples/ instead of the package root.
PKG_ROOT = next(p for p in HERE.resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(PKG_ROOT))

import band_locate as B  # noqa: E402
import datasets as CATALOG  # noqa: E402
from evalvitals.core.model import Model  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text())


# ---------------------------------------------------------------- model
#: Appended in ``mode="answer"`` so the scored tokens are an ANSWER rather than
#: the opening of a chain of thought. Kept short: every token it adds is a token
#: whose logprob enters the mean.
ANSWER_ONLY_SUFFIX = "Give only the final answer, with no explanation."


def _is_special(token: str) -> bool:
    """True for chat-control tokens like ``<|im_end|>``.

    They are dropped from the scored sequence, because a stop token is not part
    of the answer and the model is near-certain about it. Measured on this
    endpoint, keeping it moves a WRONG one-token answer from 0.629 to 0.787
    while a correct one stays at 0.996 — i.e. it compresses precisely the gap
    ``calibration`` exists to measure, and worst on the shortest answers.

    NOTE this diverges from the hf_local backend, which scores every generated
    token including EOS. Confidence numbers are therefore comparable across
    cases on this backend, but not directly against an hf_local run.
    """
    return token.startswith("<|") and token.endswith("|>")


class EndpointModel(Model):
    """The model under test, over an OpenAI-compatible endpoint.

    Subclasses ``Model`` rather than duck-typing it. Duck-typing looked fine —
    the loop mostly calls ``generate`` — and then died at M1 on
    ``model.supports(...)``, which the analyzer registry calls to decide what can
    run. Attributes alone are not the contract; inheriting also supplies
    ``unembed_weight`` and the ``call_<analyzer>`` shim, so the next thing the
    framework adds does not repeat this.

    Provides GENERATE and LOGPROBS. Sampling for ``generate`` is pinned to the
    Qwen thinking recipe because greedy decoding sends these models into verbatim
    self-verification loops that never terminate.

    Not provided: ATTENTION / HIDDEN_STATES / LOGITS. An OpenAI-compatible server
    returns text, not internals — see whitebox.py for the second-stage model that
    re-forwards a handful of cases through transformers to get those.
    """

    #: Which continuation ``logprobs`` scores. See :meth:`logprobs`.
    LOGPROBS_MODES = ("answer", "chain")

    def __init__(self, model_id: str, base_url: str, max_tokens: int, sampling: dict,
                 logprobs_mode: str = "answer", logprobs_max_tokens: int = 64,
                 logprobs_top_k: int = 5):
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
        self.modalities = frozenset({"text"})
        self.model_id = model_id
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.sampling = sampling
        self.logprobs_mode = logprobs_mode
        self.logprobs_max_tokens = logprobs_max_tokens
        self.logprobs_top_k = logprobs_top_k
        self.n_calls = 0
        self.n_truncated = 0
        self.n_logprob_calls = 0
        # M1 analyzers and the fix module's declarative candidates call
        # generate() from worker threads; the counters are read as telemetry
        # (the fix module diffs n_truncated around each candidate), so they
        # must not lose increments.
        import threading
        self._lock = threading.Lock()

    def generate(self, inputs, **kwargs) -> str:
        B.MODEL_ID, B.BASE_URL = self.model_id, self.base_url
        with self._lock:
            self.n_calls += 1
        # Portable decoding overrides a fix candidate may carry: max_tokens /
        # temperature / top_p (top_p and any other sampling key ride in the
        # sampling dict; unknown keys are ignored, never forwarded).
        sampling = {k: v for k, v in self.sampling.items() if k != "temperature"}
        if "top_p" in kwargs:
            sampling["top_p"] = kwargs["top_p"]
        text, reason = B.generate(
            str(getattr(inputs, "prompt", inputs)),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("temperature", self.sampling["temperature"]),
            sampling=sampling,
        )
        if reason == "length":
            with self._lock:
                self.n_truncated += 1
        return text

    def logprobs(self, inputs, max_new_tokens: "int | None" = None,
                 top_k: "int | None" = None, mode: "str | None" = None,
                 **kwargs) -> list:
        """Per-token logprobs of the model's own continuation.

        Signature mirrors the hf_local backend (``max_new_tokens=64, top_k=5``)
        so the same analyzers run unchanged: ``logprob_entropy`` (perplexity,
        predictive entropy) and ``calibration`` (confidence vs correctness).

        **Which continuation gets scored is the whole question for a thinking
        model, and it is why this takes a mode.** With thinking on, the first 64
        generated tokens are always the opening of a chain — "Okay, let me work
        through this" — whose probability is near-identical whether the model
        goes on to answer correctly or not. Feeding that to ``calibration``
        produces a confidence column with almost no variance, i.e. a plausible
        ECE computed on nothing.

        ``mode="answer"`` (default) sends ``enable_thinking=False`` in
        ``chat_template_kwargs``, so the template closes the think block
        immediately and the scored tokens ARE the answer. The honest caveat: the
        PASS/FAIL labels in the batch came from the model reasoning at full
        length, so this correlates no-think confidence against think-mode
        correctness. It is a proxy — a useful one, since it asks "does the model
        know this without working for it", which is exactly what separates a
        knowledge gap from a reasoning slip.

        ``mode="chain"`` scores the raw continuation with thinking left on.
        Faithful to how the batch was generated, but subject to the flatness
        above; use it to look at chain-opening entropy, not at confidence.

        Greedy (temperature 0) on purpose, unlike ``generate``: a confidence
        number that changes between calls cannot be compared across cases. The
        loop-to-the-cap failure that forbids greedy elsewhere needs thousands of
        tokens to appear; this call is capped at ~64.
        """
        import requests

        from evalvitals.core.model import TokenLogprob

        mode = (mode or self.logprobs_mode).lower()
        if mode not in self.LOGPROBS_MODES:
            raise ValueError(
                f"logprobs mode must be one of {self.LOGPROBS_MODES}, got {mode!r}")

        prompt = str(getattr(inputs, "prompt", inputs))
        if mode == "answer":
            prompt = f"{prompt}\n\n{ANSWER_ONLY_SUFFIX}"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens or self.logprobs_max_tokens),
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": int(top_k or self.logprobs_top_k),
        }
        if mode == "answer":
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        self.n_logprob_calls += 1
        last = "unknown"
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions",
                                     json=payload, timeout=600)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                entries = (choice.get("logprobs") or {}).get("content") or []
                if not entries:
                    # A server started without logprob support answers 200 with
                    # the field absent. Failing loudly beats handing the
                    # analyzers an empty list they would report as perplexity inf.
                    raise RuntimeError(
                        f"{self.base_url} returned no logprobs. vLLM supports them "
                        f"on /chat/completions; check the server is not an older "
                        f"build or a proxy that strips the field."
                    )
                return [
                    TokenLogprob(
                        token=str(e.get("token", "")),
                        logprob=float(e.get("logprob", 0.0)),
                        top={str(t["token"]): float(t["logprob"])
                             for t in (e.get("top_logprobs") or [])},
                    )
                    for e in entries
                    if not _is_special(str(e.get("token", "")))
                ]
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt == 2:
                    raise RuntimeError(f"logprobs failed after 3 attempts — {last}")
                time.sleep(2 * (attempt + 1))
        return []  # unreachable; keeps the type checker honest

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError(
            "endpoint exposes no internals — use whitebox.py (transformers) for "
            "ATTENTION / HIDDEN_STATES / LOGITS on a small selected subset"
        )

    def __repr__(self) -> str:
        return f"EndpointModel({self.model_id})"


# ---------------------------------------------------------------- inputs
def load_batch(model_id: str, dataset: str, out_dir: "Path | None" = None):
    """Load the frozen batch. *out_dir* (a tagged run dir, see ``--out-tag``) is
    read first so a smoke run is self-contained; it falls back to the untagged
    ``outputs/<model>/<dataset>/cases.json`` and copies that file into *out_dir*
    for provenance — the frozen batch itself is never rewritten."""
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    base = HERE / "outputs" / model_id / dataset / "cases.json"
    path = base
    if out_dir is not None and (out_dir / "cases.json").exists():
        path = out_dir / "cases.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run:\n"
            f"  python build_cases.py --model {model_id} --dataset {dataset}"
        )
    report = json.loads(path.read_text())
    if out_dir is not None and path == base and out_dir.resolve() != base.parent.resolve():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cases.json").write_text(path.read_text())

    # A frozen batch keeps its generations forever but its LABELS are only as
    # good as the grader that wrote them, and SKIP_STAGE0=1 reuses the file
    # wholesale. When extract_answer was fixed on 2026-08-16 the two batches on
    # disk moved 0.592->0.988 and 0.360->0.463 without a single re-generation --
    # a loop restarted on them would have mined the old bug and reported it as
    # model behaviour. Warn rather than refuse: the batch is still USABLE, it
    # just has to be regraded first, and that is free.
    try:
        from regrade import grader_fingerprint
        if report.get("grader_fingerprint") != grader_fingerprint():
            print(f"  WARNING {path.name} was graded by a different version of "
                  f"the grader than the one installed now. Its PASS/FAIL split "
                  f"may not reflect current grading.\n"
                  f"          python regrade.py --model {model_id} "
                  f"--dataset {dataset}        # see the delta\n"
                  f"          python regrade.py --model {model_id} "
                  f"--dataset {dataset} --write")
    except Exception as exc:  # never block a run on the staleness check itself
        print(f"  NOTE could not check grader freshness: {exc}")

    # Band position (build_cases writes it since 2026-08-18; older batches
    # carry only the accuracy). Out-of-band is a warning, never a stop: the
    # run is legitimate, its paired tests are just short of power.
    try:
        from build_cases import MAX_ACC, MIN_ACC, band_position
        acc = float(report.get("accuracy", 0.0))
        position = report.get("band_position") or band_position(acc)
        if position != "in":
            print(f"  WARNING batch accuracy {acc:.3f} is outside the usable band "
                  f"[{MIN_ACC}, {MAX_ACC}] ({position}: {report.get('n_fail')} FAIL / "
                  f"{report.get('n_pass')} PASS) — M2 has little to contrast and every "
                  f"paired test downstream is short of power. Results are valid but weak; "
                  f"a mid-band dataset for this model size would use the GPU hours better.")
    except Exception as exc:  # advisory only
        print(f"  NOTE could not check band position: {exc}")

    cases = [
        FailureCase(
            inputs=Inputs(prompt=c["prompt"]),
            observed=c["output"],
            expected=c["gold"] if not isinstance(c["gold"], list) else c["gold"][0],
            label=Label.PASS if c["label"] == "PASS" else Label.FAIL,
            # the grader's gold, verbatim (a list carries aliases) — see make_score_fn
            # + the generation telemetry Stage 0 recorded (finish_reason /
            # budget), which the fix module's L0 tier reads: without it a
            # truncated baseline can never be repaired by raising max_tokens.
            metadata={
                "gold": c["gold"],
                "finish_reason": c.get("finish_reason"),
                "generation_config": {"max_tokens": report.get("max_tokens")},
            },
        )
        for c in report["cases"]
    ]
    return CaseBatch(cases), report


def make_score_fn(dataset: str):
    """``(case, output) -> bool | None`` — the SAME grader that labelled the batch.

    The fix stage (and any analyzer that re-asks the model) must score a
    candidate answer by the rule the PASS/FAIL labels were written under:
    ``extract_answer`` (last ``\\boxed{}`` / ``Answer:`` span, else last line) →
    the dataset spec's grader (default ``answer_equal``, which normalises
    punctuation and case). The framework's default scorer instead checks that
    the gold string appears VERBATIM inside the whole output — on
    bbh_word_sorting a correct ``ANSWER: a, b, c`` (commas, as several fix
    candidates asked for) does not contain the gold ``a b c`` and was scored
    wrong: run5's L1/L2 candidates showed 0-2 correct of 40 (baseline 19),
    i.e. "regressed" by format, not by content. Same shape on bbh_tracking7:
    ``B`` vs gold ``(B)``.
    """
    spec = next(s for s in B.SPECS if s.name == dataset)
    grade = spec.grader or B.answer_equal
    raw = bool(getattr(spec, "grades_raw_output", False))

    def score(case, output):
        gold = (getattr(case, "metadata", None) or {}).get("gold", getattr(case, "expected", None))
        if gold is None:
            return None
        text = str(output if output is not None else "")
        graded = text if raw else B.extract_answer(text)
        try:
            return bool(grade(graded, gold))
        except Exception:
            return None

    score.__name__ = f"grade_{dataset}"
    return score


def subsample_batch(batch, report: dict, n: int, seed: int = 0):
    """Deterministic label-stratified subsample of the frozen batch (smoke runs).

    Keeps the PASS/FAIL proportion (each label rounded, at least one of each
    when both exist) so a 60-case smoke run has the same base rate as the
    full batch. Returns ``(batch, report)`` unchanged when *n* is 0 or covers
    the whole batch; the report's headline counts are recomputed so the
    printed ``[batch]`` line and summary.json describe what actually ran."""
    import random

    from evalvitals.core.case import CaseBatch, Label

    cases = list(batch)
    if n <= 0 or n >= len(cases):
        return batch, report
    by_label: dict = {}
    for c in cases:
        by_label.setdefault(c.label, []).append(c)
    rng = random.Random(seed)
    picked = []
    total = len(cases)
    for label, group in sorted(by_label.items(), key=lambda kv: str(kv[0])):
        k = max(1, round(n * len(group) / total))
        picked.extend(rng.sample(group, min(k, len(group))))
    picked = picked[:n]
    n_fail = sum(1 for c in picked if c.label == Label.FAIL)
    sub = dict(report)
    sub.update({
        "n": len(picked), "n_fail": n_fail, "n_pass": len(picked) - n_fail,
        "accuracy": (len(picked) - n_fail) / len(picked),
        "subsampled_from": len(cases), "subsample_seed": seed,
    })
    return CaseBatch(picked), sub


def build_protocol(dataset: str):
    """The human prior handed to M1/M2/M5.

    States the OBSERVATION and the grading rule only. It must NOT name a
    suspected mechanism -- proposing the mechanism is M3's job, and supplying one
    here leaks the answer into the loop that is meant to find it.
    """
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    entry = CATALOG.get(dataset)
    return ExperimentProtocol(
        description=(
            f"A text-only LLM answers items from {entry.source}. "
            f"{entry.slicing} Failure cases are items the model answered "
            f"incorrectly; the batch also contains, as controls, items from the "
            f"same slice it answered correctly. Each answer is scored against its "
            f"own gold answer."
        ),
        task_domain=entry.chapter,
        success_criteria=entry.grading,
        failure_patterns="",
        target_modalities=frozenset({"text"}),
        metadata={
            "dataset": entry.name,
            "items_in_slice": entry.items,
            "reference_accuracy_qwen35_9b": entry.accuracy_9b,
        },
    )


def build_judge(model_name: str, effort: str, timeout_sec: int = 240):
    """The judge CLI wrapper. ``timeout_sec`` bounds ONE judge call: M2's
    prompt over 10+ analyzers with a high-effort opus can exceed the wrapper's
    240 s default (bbh_causal_judgement 2026-08-18: M2 fell back to the
    threshold narrative on a 240 s timeout), so config ``judge_timeout_sec``
    raises it."""
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort, timeout_sec=int(timeout_sec))
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(
            f"judge probe: claude --model {model_name} returned empty "
            f"(rate-limited?) — try --judge-model sonnet, or a lower --judge-effort"
        )
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


#: A leading answer marker that ``extract_answer`` keeps and ``normalize_answer``
#: does not strip, so "Answer: (C)" and "The answer is (C)" would otherwise
#: normalise to "answer: c" and "answer is c" — counted as disagreement when the
#: grader scores them the same.
_ANSWER_LEAD = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answers?\s*(?:is|are)?\s*[:=-]?\s*", re.IGNORECASE)


def graded_answer(text) -> str:
    """The answer as the GRADER would see it — extracted, de-prefixed, normalised.

    Consistency should be measured in the same equivalence class the batch was
    labelled in. Anything coarser counts wording as disagreement, which on a
    model that varies its phrasing every sample is most of the signal.
    """
    from evalvitals.analyzers.reasoning._text import normalize_answer

    return normalize_answer(_ANSWER_LEAD.sub("", str(B.extract_answer(text)), count=1))


def _spends_gpu(cls) -> bool:
    """True when the analyzer's per-case loop actually calls the model.

    The cap exists to shorten wall-clock, and wall-clock is generation. Auditing
    analyzers (``arith_audit``, ``termination_audit``, ``answer_extraction_audit``)
    read the outputs already recorded in the batch — capping those would throw
    away evidence that costs nothing to collect, which is a straight loss.
    """
    import inspect

    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return True  # unknown: cap it rather than risk another straggler
    return "model.generate(" in src or "model.logprobs(" in src


def build_analyzer_overrides(max_cases: int, model=None, verbose: bool = True) -> dict:
    """Cap how many cases each per-case analyzer generates for.

    Why this exists: analyzers run in parallel with each other but iterate their
    OWN cases serially, one ``model.generate`` at a time. Against a thinking
    model on an HTTP endpoint that is ~49 s per call, so an analyzer with
    ``max_cases=128`` is a two-hour straggler that holds up the whole of M1 long
    after the other seven have finished. Measured on the bbh_tracking7 run:
    concurrency decayed 5.3 -> 1.0 and then sat at exactly 1.0 for four hours.

    This trades statistical power, not correctness: each measurement is
    unchanged, there are just fewer of them. It is deliberately reported rather
    than applied quietly, because a narrower interval that is not labelled as
    narrower is how an underpowered result gets read as a null one.

    Analyzers needing constructor arguments we cannot supply (``runs_fn``,
    ``judge``) are left alone — the loop already skips them with a warning.
    """
    import inspect

    from evalvitals.core.registry import registry

    # 0 means "library defaults" everywhere else in the config, and without this
    # it would instead cap every analyzer to ZERO cases -- analyzers that run,
    # report, and measure nothing. Silent, and it looks like a clean null result.
    if max_cases <= 0:
        return {}

    empty = inspect.Parameter.empty
    varargs = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    eligible = (set(registry.analyzers.names_compatible_with(model))
                if model is not None else None)
    overrides, capped, spared = {}, [], []
    for name, cls in sorted(registry.analyzers.all().items()):
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            continue
        spec = params.get("max_cases")
        if spec is None or not isinstance(spec.default, int):
            continue
        if spec.default <= max_cases:
            continue  # already cheaper than the cap; leave it exactly as it is
        if eligible is not None and name not in eligible:
            continue  # cannot run on this model at all
        if not _spends_gpu(cls):
            spared.append(f"{name} (reads recorded outputs, costs no generation)")
            continue
        needs = [n for n, p in params.items()
                 if n not in ("self", "max_cases")
                 and p.default is empty and p.kind not in varargs]
        if needs:
            continue
        try:
            overrides[name] = cls(max_cases=max_cases)
        except Exception:  # an analyzer that will not build is the loop's problem
            continue
        capped.append(f"{name} {spec.default}->{max_cases}")

    # self_consistency has no max_cases (it reads cases[0]) so the loop above
    # never reaches it, and its default compares WHOLE generations. On a
    # thinking model two 3k-token chains are never identical, so it reports
    # consistency = 1/n every time -- which is exactly what M2 flagged as the
    # primary anomaly on the 9B run. Comparing extracted answers fixes it.
    if "self_consistency" in (eligible if eligible is not None else {"self_consistency"}):
        try:
            sc = registry.analyzers.get("self_consistency")
            overrides["self_consistency"] = sc(answer_fn=graded_answer)
            capped.append("self_consistency compare raw_text->graded answer")
        except (KeyError, TypeError):
            pass
    if verbose:
        print(f"analyzer_max_cases={max_cases}: capped {len(capped)} analyzer(s)")
        for line in capped:
            print(f"  cap   {line}")
        for line in spared:
            print(f"  keep  {line}")
    return overrides


def load_prior_run(logs_dir: Path):
    """Reload the LAST run's M2 stats + M3 hypotheses from ``logs_dir``.

    ``run_log.jsonl`` is append-only across runs, so everything is read from
    the segment after the final ``run_start``. Returns
    ``(hypotheses, stats_report)`` — the exact hypotheses that run proposed
    and a StatsAnalysisReport carrying its per-signal tool results (with the
    BH verdicts as serialised), its multiplicity summary, its conclusion, and
    the M1 analyzer findings — everything M5, M4 and the fix module read.
    Findings objects and figures are not rebuilt (nothing downstream needs
    them). Raises SystemExit with the missing piece named when the logs do
    not hold a completed M2->M3.
    """
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.analysis.stats_tools import StatsToolResult
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.hypothesis import hypothesis_from_dict

    log_path = logs_dir / "run_log.jsonl"
    if not log_path.exists():
        raise SystemExit(f"--confirm-only: {log_path} missing — no earlier run to reload")
    events = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    starts = [i for i, e in enumerate(events) if e.get("event") == "run_start"]
    segment = events[starts[-1]:] if starts else events
    by_event = {}
    for e in segment:  # last of each kind wins inside the segment
        by_event[e.get("event")] = e

    diagnosis = by_event.get("diagnosis")
    if not diagnosis or not diagnosis.get("hypotheses"):
        raise SystemExit("--confirm-only: the last run has no M3 diagnosis with "
                         "hypotheses — nothing to confirm")
    hypotheses = [hypothesis_from_dict(h) for h in diagnosis["hypotheses"]]

    analysis = by_event.get("analysis")
    if not analysis:
        raise SystemExit("--confirm-only: the last run has no M2 analysis event")

    def _externalised(field):
        v = analysis.get(field)
        if isinstance(v, dict) and v.get("path"):
            return json.loads((logs_dir / v["path"]).read_text(encoding="utf-8"))
        return v or []

    stats_results = []
    for d in _externalised("stats_results"):
        d = dict(d)
        if d.get("ci") is not None:
            d["ci"] = tuple(d["ci"])
        stats_results.append(StatsToolResult(**d))
    if not stats_results:
        raise SystemExit("--confirm-only: the last run's M2 stats_results are empty")

    raw_results = {}
    probe = by_event.get("probe") or {}
    for name, rel in (probe.get("result_paths") or {}).items():
        path = logs_dir / rel
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        raw_results[name] = Result(analyzer=d.get("analyzer", name), model=d.get("model", ""),
                                   findings=d.get("findings") or {},
                                   metadata=d.get("metadata") or {})

    report = StatsAnalysisReport(
        model_name=diagnosis.get("model_name") or "",
        severity=analysis.get("severity") or "none",
        narrative=analysis.get("narrative") or "",
        raw_results=raw_results,
        conclusion=analysis.get("conclusion") or "",
        evidence_chain=list(analysis.get("evidence_chain") or []),
        stats_results=stats_results,
        corrected_rejections=dict(analysis.get("corrected_rejections") or {}),
        descriptive_only=bool(analysis.get("descriptive_only", False)),
    )
    return hypotheses, report


def make_scoring_note(dataset: str, cases: list) -> str:
    """One paragraph telling the fix proposer HOW an output is scored.

    The judge/coder previously had to guess the answer format from 160
    characters of prompt; this states the rule the labels were written under
    (see ``make_score_fn``) and shows a gold sample so the format is explicit.
    """
    spec = next(s for s in B.SPECS if s.name == dataset)
    grader = getattr(spec.grader, "__name__", "custom grader") if spec.grader else "answer_equal"
    raw = bool(getattr(spec, "grades_raw_output", False))
    golds = []
    for c in cases:
        g = c.get("gold")
        if g is not None and str(g) not in golds:
            golds.append(str(g))
        if len(golds) >= 3:
            break
    how = ("the WHOLE output is graded" if raw else
           "extract_answer() takes the LAST 'Answer:'-tagged span (or the last "
           "\\boxed{} / the last non-empty line when there is no tag) and grades "
           "ONLY that span")
    return (
        f"{how} with {grader}() against the gold answer (normalised: case, "
        "surrounding punctuation and '(X)' vs 'X' do not matter, wording does). "
        "The final answer must therefore appear on a final 'Answer: <answer>' line; "
        f"anything after it that looks like an answer wins. Gold looks like: "
        + " | ".join(repr(g)[:60] for g in golds)
    )


def build_codegen(backend: str):
    from evalvitals.eval_agent import CliAgentConfig

    provider = {"claude": "claude_code", "codex": "codex", "agy": "antigravity"}.get(
        backend, backend)
    is_claude = provider == "claude_code"
    effort = str(CFG.get("codegen_effort", "") or "")
    return CliAgentConfig(
        provider=provider,
        model=str(CFG.get("codegen_model", "claude-opus-5")) if is_claude else "",
        max_budget_usd=float(CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(CFG.get("codegen_timeout_sec", 480)),
        extra_args=(("--effort", effort) if (effort and is_claude) else ()),
    )


def build_explorer(codegen, out: Path):
    """The optional in-cycle explore step (free-form EDA beside the catalog M2).

    Same coder backend as M2's tool codegen, its own durable sandbox under the
    run dir so the generated ``analysis.py`` / ``tables/`` survive for audit.
    ``explore: false`` in config.yaml (or ``EXPLORE=0`` in run_all.sh) turns the
    step off; the loop then runs exactly as before.
    """
    if not bool(CFG.get("explore", True)):
        return None
    from evalvitals.agent_runtime.sandbox import ExperimentSandbox
    from evalvitals.analysis import ExploratoryAnalysisAgent

    return ExploratoryAnalysisAgent(
        cli_config=codegen,
        sandbox=ExperimentSandbox(workdir=out / "explore" / "sandbox", cleanup=False),
        timeout_sec=int(CFG.get("explore_timeout_sec", 900)),
        max_attempts=int(CFG.get("explore_max_attempts", 2)),
    )


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--base-url", default=CFG["base_url"])
    ap.add_argument("--dataset", default=CFG["dataset"])
    ap.add_argument("--judge-model", default=CFG["judge_model"])
    ap.add_argument("--judge-effort", default=CFG["judge_effort"])
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"])
    ap.add_argument("--max-cycles", type=int, default=CFG["max_cycles"])
    ap.add_argument("--confirm-split", type=float, default=CFG["confirm_split"])
    ap.add_argument("--analyzer-max-cases", type=int,
                    default=int(CFG.get("analyzer_max_cases", 0)),
                    help="cap per-analyzer case counts (0 = library defaults). "
                         "Analyzers generate serially, so this is the main knob "
                         "on M1 wall-clock; it costs power, not correctness")
    ap.add_argument("--analysis-only", action="store_true",
                    help="M1->M2->M3 and stop: propose hypotheses, skip M5 and M4")
    ap.add_argument("--fix-unverified", dest="fix_unverified", action="store_true", default=None,
                    help="when M5 verified nothing, still run the M4 intervention experiment "
                         "on the best UNVERIFIED hypothesis and then the fix stage on the "
                         "best unverified leads (flagged as unverified to the proposer). "
                         "Default: config fix_on_unverified")
    ap.add_argument("--no-fix-unverified", dest="fix_unverified", action="store_false",
                    help="require a verified hypothesis for M4 + fix (the old behaviour)")
    ap.add_argument("--skip-m4", action="store_true",
                    help="run M1->M5 but do not attempt a fix")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="label-stratified subsample of the frozen batch for a "
                         "smoke run (0 = the whole batch). Cuts M1 wall-clock; "
                         "recorded in summary.json so a wide interval reads as "
                         "'few cases', not 'no effect'")
    ap.add_argument("--out-tag", default="",
                    help="write everything to outputs/<model>/<dataset>.<tag>/ "
                         "instead of the untagged run dir (the frozen batch is "
                         "read from there or copied in) so a smoke run never "
                         "appends to a real run's logs/ or overwrites its explore/")
    ap.add_argument("--fix-validation-cases", type=int, default=None,
                    help="override config fix_validation_cases for this run (0 = the "
                         "whole confirm half). A thin-FAIL batch needs more than the "
                         "default 40 stratified cases for the paired gate to have any "
                         "power; a long-generation batch may need fewer to fit the timeout")
    ap.add_argument("--no-explore", action="store_true",
                    help="skip the in-cycle explore step (free-form EDA beside the "
                         "catalog M2) even when config.yaml has explore: true")
    ap.add_argument("--confirm-only", action="store_true",
                    help="skip M1->M3: reload the last run's M2 stats + M3 hypotheses "
                         "from outputs/<model>/<dataset>/logs/ and run M5 -> M4 -> fix "
                         "on them (logs go to logs_confirm/, summary to "
                         "summary_confirm.json)")
    args = ap.parse_args()
    if args.analysis_only and args.confirm_only:
        ap.error("--analysis-only and --confirm-only are the two halves of one run")

    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.eval_agent import (
        ExperimentWriterConfig,
        FixAgent,
        RunLogger,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent

    out = HERE / "outputs" / args.model / (
        f"{args.dataset}.{args.out_tag}" if args.out_tag else args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    batch, report_in = load_batch(args.model, args.dataset, out_dir=out)
    if args.max_cases > 0:
        batch, report_in = subsample_batch(batch, report_in, args.max_cases)
        print(f"[batch] --max-cases {args.max_cases}: label-stratified subsample "
              f"of {report_in.get('subsampled_from', '?')} frozen cases (seed 0)")

    prior = None
    if args.confirm_only:
        prior = load_prior_run(out / "logs")
        print(f"[confirm-only] reloaded {len(prior[0])} hypothesis(es) + "
              f"{len(prior[1].stats_results)} M2 tool results from {out / 'logs'}")
        for h in prior[0]:
            print(f"  - {h.statement[:110]}")

    print(f"[batch] {args.dataset} n={report_in['n']} "
          f"PASS={report_in['n_pass']} FAIL={report_in['n_fail']} "
          f"acc={report_in['accuracy']:.3f}")

    # Same budget Stage 0 used, or M1's probes truncate where the batch did not
    # and the two halves of the run stop being comparable.
    max_tokens = CATALOG.get(args.dataset).max_tokens or int(CFG["max_tokens"])
    model = EndpointModel(args.model, args.base_url, max_tokens,
                          {"temperature": float(CFG["temperature"]),
                           "top_p": float(CFG["top_p"]), "top_k": int(CFG["top_k"])},
                          logprobs_mode=str(CFG.get("logprobs_mode", "answer")),
                          logprobs_max_tokens=int(CFG.get("logprobs_max_tokens", 64)),
                          logprobs_top_k=int(CFG.get("logprobs_top_k", 5)))
    judge = build_judge(args.judge_model, args.judge_effort,
                        timeout_sec=int(CFG.get("judge_timeout_sec", 240)))
    codegen = build_codegen(args.backend)
    # A confirm-only pass never runs M1/M3, so there is nothing to explore —
    # but the analysis run's explore report is still useful to the fix
    # proposer, so it is reloaded as read-only context (explore_report=).
    explorer = (None if (args.confirm_only or args.no_explore)
                else build_explorer(codegen, out))
    explore_report = None
    if args.confirm_only:
        prior_explore = out / "explore" / "exploratory_report.json"
        if prior_explore.exists():
            try:
                explore_report = json.loads(prior_explore.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                explore_report = None
    overrides = (build_analyzer_overrides(args.analyzer_max_cases, model=model)
                 if args.analyzer_max_cases > 0 else {})
    # A confirm-only pass logs beside the analysis it reuses, never over it —
    # the dashboard merges every logs*/run_log.jsonl under the run dir.
    logger = RunLogger(run_dir=out / ("logs_confirm" if args.confirm_only else "logs"),
                       verbose=True)

    # max_cases_per_analyzer is the BACKSTOP for analyzer_overrides: the
    # overrides can only turn a `max_cases` constructor knob, and the five
    # analyzers that dominated M1 wall-clock here (first_error_judge, rise,
    # self_consistency, verbalized_confidence, trajectory_rubric) expose none.
    # Without it the cap reached 7 of 20 generating analyzers and the rest ran
    # the full batch one generate() at a time.
    probe_agent = ProbeAgent(judge=judge, allow_codegen=True,
                             codegen_config=codegen,
                             analyzer_overrides=overrides,
                             max_cases_per_analyzer=args.analyzer_max_cases)

    # Every stage takes its judge/coder through its CONSTRUCTOR. Assigning
    # `stage.judge` afterwards would leave each stage on its own default and the
    # run would complete looking normal while none of the configured judge
    # reached M1/M2/M3/M5.
    loop = VLDiagnoseLoop(
        model=model,
        protocol=build_protocol(args.dataset),
        # max_cases_per_analyzer is the BACKSTOP for analyzer_overrides: the
        # overrides can only turn a `max_cases` constructor knob, and the five
        # analyzers that dominated M1 wall-clock here (first_error_judge, rise,
        # self_consistency, verbalized_confidence, trajectory_rubric) expose
        # none. Without it the cap reached 7 of 20 generating analyzers and the
        # rest ran the full batch serially.
        probe_agent=probe_agent,
        # figure_dir: the catalog M2's forest plot (effect +- CI per tool)
        # lands in logs/figures/m2_effects.png and is listed in the analysis
        # event's `figures`; without it M2 stays JSON-only.
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen,
                                       figure_dir=str(logger.run_dir / "figures")),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(
            judge=judge,
            max_tier=str(CFG.get("fix_max_tier", "L3b")),
            cli_config=codegen,
            run_logger=logger,
            # Score candidates with the batch's own grader, not the framework's
            # verbatim-substring default (see make_score_fn).
            score_fn=make_score_fn(args.dataset),
            max_validation_cases=(args.fix_validation_cases
                                  if args.fix_validation_cases is not None
                                  else int(CFG.get("fix_validation_cases", 0))),
            exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 1800)),
            # The baseline's decoding budget is the FLOOR for every candidate:
            # a judge-proposed max_tokens below it is raised to it (tracking7's
            # judge set 900 against a 4096 baseline whose median output was 806
            # tokens -> "regressed" by truncation). Also shown to the proposer.
            baseline_generation_kwargs={"max_tokens": max_tokens,
                                        "temperature": float(CFG["temperature"]),
                                        "top_p": float(CFG["top_p"])},
            # Declarative candidates run their cases in parallel against the
            # endpoint (coded pipelines stay serial through the bridge).
            concurrency=int(CFG.get("fix_concurrency", 1)),
            # Noise model: the baseline is a per-case PASS RATE (frozen sample
            # + k-1 fresh at the batch's own T), the paired test runs on rate
            # differences (betting e-value) — sampling-unstable cases are
            # weighed, not dropped, and one T=0.6 sample per arm no longer
            # decides fixed/broken. Candidates default to one pass (coded
            # pipelines can't repeat cheaply); raise fix_candidate_repeats to
            # average template/spec candidates too.
            baseline_repeats=int(CFG.get("fix_baseline_repeats", 1)),
            candidate_repeats=int(CFG.get("fix_candidate_repeats", 1)),
            scoring_note=make_scoring_note(args.dataset, report_in.get("cases") or []),
            floor_candidates=tuple(CFG.get("fix_floor_candidates",
                                           ["self_consistency_5"]) or ()),
        ),
        max_cycles=args.max_cycles,
        run_logger=logger,
        confirm_split=args.confirm_split,
        # Explore beside the catalog M2, not instead of it: a free-form EDA pass
        # over the same M1 per-case table, between M1 and M2. Its
        # observations/charts reach M3 as UNCONFIRMED notes and land under
        # outputs/<model>/<dataset>/explore/ (exploratory_report.json + tables/
        # + figures/) for the dashboard. M2's confirmatory family, M5 and the
        # fix gate never see it.
        explorer=explorer,
        explore_dir=out / "explore",
        explore_report=explore_report,
    )

    if args.analysis_only:
        report = loop.run_analysis(batch)
        print(f"[M1-M3] proposed {len(report.final_hypotheses)} hypotheses")
    else:
        if args.confirm_only:
            hypotheses, stats_report = prior
            report = loop.run_confirm(batch, hypotheses, stats_report=stats_report)
            print(f"[M5 confirm-only] stopped_by={report.stopped_by} "
                  f"verified={len(report.verified_hypotheses)}/"
                  f"{len(report.all_test_results)}")
        else:
            report = loop.run(batch)
            print(f"[M1-M5] cycles={report.cycles} stopped_by={report.stopped_by} "
                  f"verified={len(report.verified_hypotheses)}/"
                  f"{len(report.all_test_results)}")
        for t in report.all_test_results:
            stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
            print(f"  - [{t.status}] conf={t.confidence:.2f} "
                  f"grade={t.evidence_grade} {stmt[:100]}")
        if not args.skip_m4:
            fix_unverified = (args.fix_unverified if args.fix_unverified is not None
                              else bool(CFG.get("fix_on_unverified", True)))
            fix = loop.run_m4(report, batch, allow_unverified=fix_unverified)
            if fix is not None:
                tag = "verified" if report.verified_hypotheses else "UNVERIFIED (best lead)"
                print(f"[M4] experiment run on the {tag} hypothesis: status={fix.status}")
            else:
                print("[M4] no hypothesis to experiment on"
                      + ("" if fix_unverified else " (no verified hypothesis; "
                         "--fix-unverified / fix_on_unverified to proceed anyway)"))
            if fix is not None or (fix_unverified and report.final_hypotheses):
                if not report.verified_hypotheses:
                    print("[fix] no verified hypothesis — proposing on the best unverified "
                          "leads (flagged to the proposer); the candidate validation decides")
                outcome = loop.run_fix(report, batch)
                print("[fix]", getattr(outcome, "recommendation", None) or outcome)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "batch": {k: report_in[k] for k in
                  ("n", "accuracy", "n_pass", "n_fail", "truncated_rate")},
        "confirm_split": args.confirm_split,
        "analysis_only": args.analysis_only,
        "confirm_only": args.confirm_only,
        "explore": explorer is not None,
        "max_cases": args.max_cases or None,
        "out_tag": args.out_tag or None,
        "fix_validation_cases": (args.fix_validation_cases
                                 if args.fix_validation_cases is not None
                                 else int(CFG.get("fix_validation_cases", 0))),
        "fix_baseline_repeats": int(CFG.get("fix_baseline_repeats", 1)),
        "fix_on_unverified": (args.fix_unverified if args.fix_unverified is not None
                              else bool(CFG.get("fix_on_unverified", True))),
        "fix_candidate_repeats": int(CFG.get("fix_candidate_repeats", 1)),
        "n_hypotheses": len(getattr(report, "hypotheses", None)
                            or getattr(report, "all_hypotheses", None) or []),
        "n_verified": len(getattr(report, "verified_hypotheses", []) or []),
        "model_calls": model.n_calls,
        "model_truncated": model.n_truncated,
        "logprob_calls": model.n_logprob_calls,
        "logprobs_mode": model.logprobs_mode,
        # recorded so a narrow interval downstream is readable as "fewer cases",
        # not as "no effect"
        "analyzer_max_cases": args.analyzer_max_cases or None,
        # two different mechanisms, and the gap between them is the point:
        # `capped` turned a constructor knob, `truncated` bounded the analyzers
        # that have no knob to turn (name -> [cases offered, cases used]).
        "analyzers_capped": sorted(overrides),
        "analyzers_truncated": {k: list(v) for k, v
                                in sorted(probe_agent.capped_analyzers.items())},
    }
    summary_name = "summary_confirm.json" if args.confirm_only else "summary.json"
    (out / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/summary_name}")
    print(f"dashboard: python -m evalvitals.cli dashboard {out}")


if __name__ == "__main__":
    main()
