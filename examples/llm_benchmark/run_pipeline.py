"""M1 -> M2 -> M3 -> M5 -> M4 over a frozen CaseBatch from build_cases.py.

    python run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law
    python run_pipeline.py --analysis-only        # M1->M2->M3, stop before M5/M4
    python -m evalvitals.cli dashboard outputs/qwen3.5-9b/supergpqa_law

Stages (evalvitals.eval_agent.loop.VLDiagnoseLoop -- the class name says VL, but
it takes a plain Model plus an ExperimentProtocol whose target_modalities is
{"text"} here, and nothing in it is vision-specific):

    M1 ProbeAgent          selects and runs analyzers against the batch
    M2 StatsAnalysisAgent  protocol-aware statistics over M1's per-case signals
    M3 DiagnosisAgent      proposes hypotheses from the stats
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
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))
sys.path.insert(0, str(HERE.parent.parent))

import band_locate as B  # noqa: E402
import datasets as CATALOG  # noqa: E402

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


class EndpointModel:
    """The model under test, over an OpenAI-compatible endpoint.

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

    def generate(self, inputs, **kwargs) -> str:
        B.MODEL_ID, B.BASE_URL = self.model_id, self.base_url
        self.n_calls += 1
        text, reason = B.generate(
            str(getattr(inputs, "prompt", inputs)),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("temperature", self.sampling["temperature"]),
            sampling={k: v for k, v in self.sampling.items() if k != "temperature"},
        )
        if reason == "length":
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
def load_batch(model_id: str, dataset: str):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = HERE / "outputs" / model_id / dataset / "cases.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run:\n"
            f"  python build_cases.py --model {model_id} --dataset {dataset}"
        )
    report = json.loads(path.read_text())
    cases = [
        FailureCase(
            inputs=Inputs(prompt=c["prompt"]),
            observed=c["output"],
            expected=c["gold"] if not isinstance(c["gold"], list) else c["gold"][0],
            label=Label.PASS if c["label"] == "PASS" else Label.FAIL,
        )
        for c in report["cases"]
    ]
    return CaseBatch(cases), report


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


def build_judge(model_name: str, effort: str):
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(
            f"judge probe: claude --model {model_name} returned empty "
            f"(rate-limited?) — try --judge-model sonnet, or a lower --judge-effort"
        )
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


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
    ap.add_argument("--analysis-only", action="store_true",
                    help="M1->M2->M3 and stop: propose hypotheses, skip M5 and M4")
    ap.add_argument("--skip-m4", action="store_true",
                    help="run M1->M5 but do not attempt a fix")
    args = ap.parse_args()

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

    batch, report_in = load_batch(args.model, args.dataset)
    out = HERE / "outputs" / args.model / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"[batch] {args.dataset} n={report_in['n']} "
          f"PASS={report_in['n_pass']} FAIL={report_in['n_fail']} "
          f"acc={report_in['accuracy']:.3f}")

    model = EndpointModel(args.model, args.base_url, CFG["max_tokens"],
                          {"temperature": float(CFG["temperature"]),
                           "top_p": float(CFG["top_p"]), "top_k": int(CFG["top_k"])},
                          logprobs_mode=str(CFG.get("logprobs_mode", "answer")),
                          logprobs_max_tokens=int(CFG.get("logprobs_max_tokens", 64)),
                          logprobs_top_k=int(CFG.get("logprobs_top_k", 5)))
    judge = build_judge(args.judge_model, args.judge_effort)
    codegen = build_codegen(args.backend)
    logger = RunLogger(run_dir=out / "logs", verbose=True)

    # Every stage takes its judge/coder through its CONSTRUCTOR. Assigning
    # `stage.judge` afterwards would leave each stage on its own default and the
    # run would complete looking normal while none of the configured judge
    # reached M1/M2/M3/M5.
    loop = VLDiagnoseLoop(
        model=model,
        protocol=build_protocol(args.dataset),
        probe_agent=ProbeAgent(judge=judge, allow_codegen=True,
                               codegen_config=codegen),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(
            judge=judge,
            max_tier=str(CFG.get("fix_max_tier", "L3b")),
            cli_config=codegen,
            run_logger=logger,
            max_validation_cases=int(CFG.get("fix_validation_cases", 0)),
            exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 1800)),
        ),
        max_cycles=args.max_cycles,
        run_logger=logger,
        confirm_split=args.confirm_split,
    )

    if args.analysis_only:
        report = loop.run_analysis(batch)
        print(f"[M1-M3] proposed {len(report.hypotheses)} hypotheses")
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
            fix = loop.run_m4(report, batch)
            print(f"[M4] {'fix proposed' if fix else 'no verified hypothesis to fix'}")
            if fix is not None:
                outcome = loop.run_fix(report, batch)
                print("[fix]", getattr(outcome, "recommendation", None) or outcome)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "batch": {k: report_in[k] for k in
                  ("n", "accuracy", "n_pass", "n_fail", "truncated_rate")},
        "confirm_split": args.confirm_split,
        "analysis_only": args.analysis_only,
        "n_hypotheses": len(getattr(report, "hypotheses", []) or []),
        "n_verified": len(getattr(report, "verified_hypotheses", []) or []),
        "model_calls": model.n_calls,
        "model_truncated": model.n_truncated,
        "logprob_calls": model.n_logprob_calls,
        "logprobs_mode": model.logprobs_mode,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/'summary.json'}")
    print(f"dashboard: python -m evalvitals.cli dashboard {out}")


if __name__ == "__main__":
    main()
