"""Contract checks for the two capability stages of examples/dataset_selection/llm_benchmark.

The failures these exist to catch are all SILENT ones — each would produce a
plausible number rather than an error:

  * scoring the chat stop token, which drags every short answer toward 1.0 and
    flattens exactly the PASS/FAIL gap `calibration` is there to measure;
  * sending `enable_thinking=False` in chain mode (or forgetting it in answer
    mode), which changes what the confidence number means without changing its
    shape;
  * assuming a dense stack on Qwen3.5, where 24 of 32 layers are linear
    attention and the captured list is 8 long with positions that are not layer
    numbers;
  * an unbalanced Stage W subset, which would run the PASS/FAIL contrast at a
    third of the power it reports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "examples" / "dataset_selection" / "llm_benchmark")


def _load(name: str):
    pytest.importorskip("requests")
    pytest.importorskip("yaml")
    sys.path.insert(0, str(_BENCH))
    sys.path.insert(0, str(_BENCH.parent / "llm_band_probe"))
    try:
        spec = importlib.util.spec_from_file_location(name, _BENCH / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for p in (str(_BENCH), str(_BENCH.parent / "llm_band_probe")):
            if p in sys.path:
                sys.path.remove(p)


@pytest.fixture(scope="module")
def pipe():
    return _load("run_pipeline")


@pytest.fixture(scope="module")
def wb():
    return _load("whitebox")


@pytest.fixture(scope="module")
def runner():
    return _load("run_whitebox")


# ── logprobs over the endpoint ───────────────────────────────────────────────
class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _reply(tokens):
    return {"choices": [{"logprobs": {"content": [
        {"token": t, "logprob": lp, "top_logprobs": [{"token": t, "logprob": lp}]}
        for t, lp in tokens
    ]}}]}


def _model(pipe, **kw):
    return pipe.EndpointModel("m", "http://x/v1", 2048,
                              {"temperature": 0.6, "top_p": 0.95, "top_k": 20}, **kw)


def test_stop_token_is_not_scored(pipe, monkeypatch):
    """The EOS token is near-certain and is not part of the answer.

    Measured live: keeping it moved a wrong one-token answer from 0.629 to
    0.787 while a correct one stayed at 0.996 — i.e. it eats the gap.
    """
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_reply([("9", -0.46), ("<|im_end|>", -0.015)])))
    toks = _model(pipe).logprobs("q")
    assert [t.token for t in toks] == ["9"]


def test_ordinary_angle_bracket_text_is_kept(pipe, monkeypatch):
    """Only chat-control tokens are dropped — not maths or code that looks like one."""
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp(_reply([("<", -0.1), ("|", -0.2), ("x", -0.3)])))
    assert [t.token for t in _model(pipe).logprobs("q")] == ["<", "|", "x"]


def test_answer_mode_disables_thinking_and_chain_mode_does_not(pipe, monkeypatch):
    import requests

    seen = {}

    def _capture(url, json=None, **k):
        seen.clear()
        seen.update(json)
        return _Resp(_reply([("a", -0.1)]))

    monkeypatch.setattr(requests, "post", _capture)

    _model(pipe, logprobs_mode="answer").logprobs("2+2?")
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert pipe.ANSWER_ONLY_SUFFIX in seen["messages"][0]["content"]
    assert seen["temperature"] == 0.0, "a confidence number must be reproducible"
    assert seen["logprobs"] is True

    _model(pipe, logprobs_mode="chain").logprobs("2+2?")
    assert "chat_template_kwargs" not in seen
    assert pipe.ANSWER_ONLY_SUFFIX not in seen["messages"][0]["content"]


def test_unknown_mode_is_rejected_before_any_call(pipe, monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("must not reach the network")

    monkeypatch.setattr(requests, "post", _boom)
    with pytest.raises(ValueError, match="mode"):
        _model(pipe, logprobs_mode="whatever").logprobs("q")


def test_missing_logprobs_field_raises_rather_than_returning_empty(pipe, monkeypatch):
    """An empty list would be reported downstream as perplexity inf, not as a bug."""
    import requests

    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp({"choices": [{"logprobs": None}]}))
    monkeypatch.setattr(pipe.time, "sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="no logprobs"):
        _model(pipe).logprobs("q")


def test_model_declares_logprobs(pipe):
    from evalvitals.core.capability import Capability

    caps = _model(pipe).capabilities
    assert Capability.LOGPROBS in caps and Capability.GENERATE in caps
    assert Capability.ATTENTION not in caps, "the endpoint cannot provide internals"


def test_forward_still_refuses_and_says_where_to_go(pipe):
    with pytest.raises(NotImplementedError, match="whitebox"):
        _model(pipe).forward("q", capture=set())


def test_endpoint_model_satisfies_the_registry(pipe):
    """The regression that killed the first full run.

    ``EndpointModel`` carried `capabilities` and `modalities` but was not a
    ``Model``, so it had no ``supports()``. Everything looked right until M1
    asked the registry what could run and got an AttributeError — after Stage 0
    had already spent four hours on the GPU.
    """
    from evalvitals.core.model import Model as _Model
    from evalvitals.core.registry import registry

    model = _model(pipe)
    assert isinstance(model, _Model)
    names = registry.analyzers.names_compatible_with(model)
    assert "logprob_entropy" in names and "calibration" in names
    assert "attention_sink" not in names, "the endpoint has no internals"


# ── analyzer case caps ───────────────────────────────────────────────────────
def test_cap_only_lowers_never_raises(pipe):
    """A cap must not turn a cheap analyzer into an expensive one."""
    import inspect

    from evalvitals.core.registry import registry

    overrides = pipe.build_analyzer_overrides(32, model=_model(pipe), verbose=False)
    for name, instance in overrides.items():
        params = inspect.signature(registry.analyzers.get(name).__init__).parameters
        if "max_cases" not in params:
            continue  # not a cap — e.g. the self_consistency answer_fn override
        default = params["max_cases"].default
        assert default > 32, f"{name} defaulted to {default}, should have been left alone"
        assert instance.max_cases == 32


def test_self_consistency_is_overridden_to_compare_answers(pipe):
    """Its default compares whole generations, which on a thinking model reports
    consistency = 1/n no matter what the model answered. That artifact was the
    only 'anomaly' M2 found on the first full 9B run."""
    overrides = pipe.build_analyzer_overrides(32, model=_model(pipe), verbose=False)
    sc = overrides.get("self_consistency")
    assert sc is not None and sc.answer_fn is not None
    # measured in the grader's equivalence class, so wording is not disagreement
    assert sc.answer_fn("reasoning...\nAnswer: (C)") == sc.answer_fn("blah\nThe answer is (C)")
    assert sc.answer_fn("x\nAnswer: (A)") != sc.answer_fn("x\nAnswer: (B)")


def test_cap_skips_analyzers_the_model_cannot_run(pipe):
    """White-box analyzers must not be instantiated for an endpoint model."""
    overrides = pipe.build_analyzer_overrides(32, model=_model(pipe), verbose=False)
    for name in ("logit_lens", "linear_probe", "layer_contrast"):
        assert name not in overrides


def test_cap_skips_analyzers_needing_constructor_args(pipe):
    """reliability_probe needs runs_fn; building it blind would raise inside M1."""
    overrides = pipe.build_analyzer_overrides(32, model=_model(pipe), verbose=False)
    for name in ("reliability_probe", "tool_shap", "trajectory_rubric", "chair"):
        assert name not in overrides


def test_generation_detection_is_read_from_source_not_assumed(pipe):
    """arith_audit LOOKS like a pure audit of recorded outputs and is not.

    It re-asks the model, so it belongs under the cap. Assuming otherwise would
    have left a 200-case straggler in place.
    """
    from evalvitals.core.registry import registry

    assert pipe._spends_gpu(registry.analyzers.get("arith_audit"))
    assert pipe._spends_gpu(registry.analyzers.get("termination_audit"))


def test_zero_means_library_defaults(pipe):
    assert pipe.build_analyzer_overrides(0, model=_model(pipe), verbose=False) == {}


# ── Stage W ──────────────────────────────────────────────────────────────────
class _FakeInner:
    """Minimal hf_local stand-in: config shape + a forward that records its spec."""

    def __init__(self, layer_types, heads=16):
        class _Text:
            pass

        text = _Text()
        text.layer_types = layer_types
        text.num_attention_heads = heads
        text.num_hidden_layers = len(layer_types)

        class _Conf:
            pass

        conf = _Conf()
        conf.text_config = text

        class _HF:
            pass

        hf = _HF()
        hf.config = conf
        self._loaded = (hf, _FakeTok())
        self.calls = []

    def forward(self, inputs, capture, spec=None):
        self.calls.append(spec)
        return "trace"


class _FakeTok:
    def __call__(self, text):
        return {"input_ids": list(range(max(1, len(str(text)) // 4)))}


_HYBRID = ["linear_attention"] * 3 + ["full_attention"]


def test_hybrid_stack_reports_capturable_layers_not_depth(wb):
    inner = _FakeInner(_HYBRID * 8)
    m = wb.BoundedWhitebox(inner)
    assert m.attention_layers() == [3, 7, 11, 15, 19, 23, 27, 31]
    assert m._shape() == (8, 16), "budget must be sized on the 8 that exist, not 32"


def test_dense_stack_still_reports_every_layer(wb):
    m = wb.BoundedWhitebox(_FakeInner(["full_attention"] * 12))
    assert m.attention_layers() == list(range(12))


def test_missing_layer_types_falls_back_to_depth(wb):
    m = wb.BoundedWhitebox(_FakeInner([]))
    m._loaded[0].config.text_config.num_hidden_layers = 5
    assert m.attention_layers() == [0, 1, 2, 3, 4]


def test_forward_refuses_past_the_budget_with_an_actionable_message(wb):
    from evalvitals.core.capability import Capability

    m = wb.BoundedWhitebox(_FakeInner(_HYBRID * 8), budget_bytes=1024)
    with pytest.raises(MemoryError) as exc:
        m.forward("x" * 4000, capture={Capability.ATTENTION})
    text = str(exc.value)
    assert "GB" in text and "budget" in text
    assert "rollout" in text, "the layer-subset escape hatch must carry its caveat"


def test_budget_only_guards_attention(wb):
    """Hidden states are O(seq x dim), not O(seq^2) — the guard must not block them."""
    from evalvitals.core.capability import Capability

    inner = _FakeInner(_HYBRID * 8)
    m = wb.BoundedWhitebox(inner, budget_bytes=1)
    m.forward("x" * 4000, capture={Capability.HIDDEN_STATES})
    assert len(inner.calls) == 1


def test_forward_injects_a_default_capture_spec(wb):
    """Analyzers call forward() with no spec; without this every capture is unbounded."""
    from evalvitals.core.capability import Capability

    inner = _FakeInner(_HYBRID * 8)
    m = wb.BoundedWhitebox(inner, layers=[0, 7], to_cpu=True)
    m.forward("short", capture={Capability.ATTENTION})
    assert inner.calls[0].layers == [0, 7]
    assert inner.calls[0].to_cpu is True


def test_caller_supplied_spec_wins(wb):
    from evalvitals.core.capability import Capability
    from evalvitals.core.model import CaptureSpec

    inner = _FakeInner(_HYBRID * 8)
    m = wb.BoundedWhitebox(inner, layers=[0])
    m.forward("short", capture={Capability.ATTENTION}, spec=CaptureSpec(layers=[3]))
    assert inner.calls[0].layers == [3]


def test_attention_bytes_is_quadratic_in_sequence(wb):
    assert wb.attention_bytes(2048, 8, 16) == 4 * wb.attention_bytes(1024, 8, 16)


# ── subset selection ─────────────────────────────────────────────────────────
def _report(n_pass, n_fail, prompt="q" * 40):
    return {"cases": (
        [{"label": "PASS", "prompt": prompt, "output": "o", "gold": "g"}] * n_pass
        + [{"label": "FAIL", "prompt": prompt, "output": "o", "gold": "g"}] * n_fail
    )}


def test_subset_is_label_balanced_even_when_the_batch_is_not(wb):
    """A 0.70-accuracy slice would otherwise hand FAIL a third of the power."""
    sel = wb.select_cases(_report(70, 30), n=20, seed=0)
    assert sum(1 for c in sel if c["label"] == "PASS") == 10
    assert sum(1 for c in sel if c["label"] == "FAIL") == 10


def test_subset_is_deterministic_per_seed(wb):
    rep = _report(40, 40)
    a = wb.select_cases(rep, n=8, seed=3)
    assert [c["label"] for c in a] == [c["label"] for c in wb.select_cases(rep, n=8, seed=3)]


def test_long_prompts_are_dropped_before_selection(wb):
    rep = {"cases": [
        {"label": "PASS", "prompt": "x" * 40, "output": "o", "gold": "g"},
        {"label": "PASS", "prompt": "x" * 4000, "output": "o", "gold": "g"},
        {"label": "FAIL", "prompt": "x" * 40, "output": "o", "gold": "g"},
        {"label": "FAIL", "prompt": "x" * 4000, "output": "o", "gold": "g"},
    ]}
    sel = wb.select_cases(rep, n=4, seed=0, max_prompt_tokens=20, tokenizer=_FakeTok())
    assert len(sel) == 2 and all(len(c["prompt"]) == 40 for c in sel)


# ── the PASS/FAIL contrast ───────────────────────────────────────────────────
def test_contrast_reports_direction_and_both_ns(runner):
    rows = ([{"label": "FAIL", "s": 0.4}] * 4) + ([{"label": "PASS", "s": 0.2}] * 4)
    c = runner.contrast(rows, "s")
    assert c["n_pass"] == 4 and c["n_fail"] == 4
    assert c["gap"] == pytest.approx(0.2), "gap is FAIL minus PASS"


def test_contrast_withholds_effect_size_when_a_side_is_too_thin(runner):
    rows = [{"label": "FAIL", "s": 0.4}, {"label": "PASS", "s": 0.2},
            {"label": "PASS", "s": 0.3}]
    c = runner.contrast(rows, "s")
    assert "gap" not in c and c["n_fail"] == 1


def test_contrast_survives_zero_variance(runner):
    rows = ([{"label": "FAIL", "s": 0.4}] * 3) + ([{"label": "PASS", "s": 0.4}] * 3)
    assert runner.contrast(rows, "s")["cohens_d"] is None


def test_rollout_is_not_a_default_analyzer(runner):
    """It composes through the stack; on a hybrid model it sees 8 layers of 32."""
    assert "attention_rollout" not in runner.DEFAULT_ANALYZERS
    assert "attention_rollout" in runner.DEPTH_COMPOSING


def test_booleans_are_not_averaged_as_numbers(runner):
    assert runner.numeric_findings({"ok": True, "x": 2, "name": "s"}) == {"x": 2.0}


# ── the spec ─────────────────────────────────────────────────────────────────
def test_qwen35_specs_declare_the_hybrid_stack():
    from evalvitals.core.spec import AttnSemantics
    from evalvitals.specs import get_spec

    for key in ("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"):
        spec = get_spec(key)
        assert spec.attn_semantics is AttnSemantics.HYBRID_SPARSE
        assert spec.is_reasoning, "thinking is on by default for these checkpoints"
        assert any("full_attention_interval" in c for c in spec.caveats)


def test_hybrid_sparse_still_grants_the_attention_capability():
    """It is not NONE: the 8 tensors are real, dense and worth reading."""
    from evalvitals.core.spec import AttnSemantics

    assert AttnSemantics.HYBRID_SPARSE is not AttnSemantics.NONE


# ── regrade: a frozen batch's labels go stale, its generations do not ────────
@pytest.fixture(scope="module")
def rg():
    return _load("regrade")


def _batch(dataset="bbh_tracking7", cases=()):
    return {
        "model": "qwen3.5-9b", "dataset": dataset, "n": len(cases),
        "accuracy": sum(c["label"] == "PASS" for c in cases) / max(len(cases), 1),
        "cases": list(cases),
    }


def _rg_case(output, gold, label, truncated=False):
    return {"prompt": "q", "gold": gold, "output": output, "label": label,
            "finish_reason": "length" if truncated else "stop",
            "truncated": truncated}


def test_regrade_recovers_a_bare_option_label(rg):
    """The exact shape that cost the 9B batch 99 cases."""
    report = _batch(cases=[
        _rg_case("Claire is dancing with **Lola**.\n\nAnswer: (A)", "(A)", "FAIL"),
        _rg_case("Answer: (B)", "(B)", "PASS"),
    ])
    delta = rg.regrade(report)
    assert delta["fail_to_pass"] == 1
    assert delta["pass_to_fail"] == 0
    assert delta["labels"] == ["PASS", "PASS"]
    assert delta["new_accuracy"] == 1.0


def test_regrade_does_not_mutate_its_input(rg):
    """Dry run is the default, so the report must survive being inspected."""
    report = _batch(cases=[_rg_case("Answer: (A)", "(A)", "FAIL")])
    before = json.dumps(report, sort_keys=True)
    rg.regrade(report)
    assert json.dumps(report, sort_keys=True) == before


def test_regrade_counts_recovered_truncated_cases_separately(rg):
    """A cut-off generation that happens to regrade PASS is budget noise."""
    report = _batch(cases=[_rg_case("Answer: (A)", "(A)", "FAIL", truncated=True)])
    delta = rg.regrade(report)
    assert delta["fail_to_pass"] == 1
    assert delta["fail_to_pass_truncated"] == 1


def test_apply_writes_labels_accuracy_and_the_fingerprint(rg):
    report = _batch(cases=[
        _rg_case("Answer: (A)", "(A)", "FAIL"),
        _rg_case("Answer: (C)", "(B)", "PASS"),
    ])
    out = rg.apply(report, rg.regrade(report))
    assert [c["label"] for c in out["cases"]] == ["PASS", "FAIL"]
    assert out["accuracy"] == 0.5
    assert out["n_pass"] == 1 and out["n_fail"] == 1
    assert out["grader_fingerprint"] == rg.grader_fingerprint()


def test_fingerprint_tracks_the_grading_source_not_a_version_constant(rg):
    """Hand-bumped versions record only the changes someone remembered to."""
    from evalvitals.analyzers.reasoning import _text

    digest = rg.grader_fingerprint()
    assert digest == rg.grader_fingerprint(), "must be deterministic"
    assert len(digest) == 16
    # it really is derived from that file's bytes
    import hashlib
    assert digest == hashlib.sha256(
        Path(_text.__file__).read_bytes()).hexdigest()[:16]


def test_a_freshly_built_batch_is_not_reported_stale():
    """build_cases must stamp the same fingerprint run_pipeline checks."""
    build_cases = _load("build_cases")
    regrade = _load("regrade")
    assert build_cases._grader_fingerprint() == regrade.grader_fingerprint()
