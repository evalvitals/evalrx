"""M1 white-box tier(b): capture-then-compute attention probe generation.

A scripted "LLM" returns a real numpy probe over the captured attention dump,
so the full capture -> generate -> sandbox-run -> parse path runs
deterministically (no GPU, no network, no real coding agent).
"""

from __future__ import annotations

import torch

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model, Trace
from evalrx.eval_agent import WhiteboxProbeGenerator

_SEQ = 6
_IMG = slice(2, 5)  # 3 image tokens; position 0 is a structural token


class AttnVLM(Model):
    """Fake VLM. FAIL-prompt cases dump attention onto the structural token at
    position 0; PASS-prompt cases spread attention over the image region."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "x"

    def forward(self, inputs, capture, spec=None) -> Trace:
        prompt = str(getattr(inputs, "prompt", inputs))
        row = torch.full((_SEQ,), 0.02)
        if "FAIL" in prompt:
            row[0] = 0.7          # structural-token sink
        else:
            row[_IMG] = 0.3       # attends to image patches
        layer = torch.zeros(2, _SEQ, _SEQ)
        layer[:, -1, :] = row
        mask = torch.zeros(_SEQ, dtype=torch.bool)
        mask[_IMG] = True
        toks = ["<|im_start|>", "user", "img", "img", "img", "?"]
        return Trace(
            tokens=toks, token_ids=list(range(_SEQ)),
            provided={Capability.ATTENTION},
            attentions=[layer.clone() for _ in range(3)],
            extras={"image_token_mask": mask},
        )


class NoAttnModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "x"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


# Probe: mass on the structural token at position 0, per case.
_STRUCT_SINK_PROBE = '''
import json, numpy as np
man = json.load(open("m1_whitebox_manifest.json"))
per_case, vals = [], []
for c in man["cases"]:
    d = np.load(f"m1_whitebox/{c['id']}.npz")
    attn = d["attn_last"]                 # (n_layers, seq)
    struct = float(attn[:, 0].mean())     # position 0 is <|im_start|>
    vals.append(struct)
    per_case.append({"sample_id": c["id"], "struct_sink_mass": round(struct, 4)})
print("PROBE_RESULT_JSON=" + json.dumps({
    "findings": {"mean_struct_sink_mass": round(float(np.mean(vals)), 4)},
    "per_case": per_case,
}))
'''

_NO_MARKER = 'print("nothing")'


class ScriptedJudge(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, inputs, **kwargs) -> str:
        return f"```python\n{self._code}\n```"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _batch() -> CaseBatch:
    return CaseBatch([
        FailureCase(id="p0", inputs=Inputs(prompt="PASS q", image="<img>"), label=Label.PASS),
        FailureCase(id="p1", inputs=Inputs(prompt="PASS r", image="<img>"), label=Label.PASS),
        FailureCase(id="f0", inputs=Inputs(prompt="FAIL q", image="<img>"), label=Label.FAIL),
        FailureCase(id="f1", inputs=Inputs(prompt="FAIL r", image="<img>"), label=Label.FAIL),
    ])


def test_capture_compute_produces_per_case_signal():
    gen = WhiteboxProbeGenerator(judge=ScriptedJudge(_STRUCT_SINK_PROBE))
    result, probe = gen.generate("structural token attention sink", AttnVLM(), _batch(),
                                 name="struct_sink")
    assert result is not None and probe is not None
    assert result.analyzer == "generated_wb:struct_sink"
    by_id = {e["sample_id"]: e["struct_sink_mass"] for e in result.findings["per_case"]}
    # FAIL cases sink ~0.7 on the structural token; PASS cases ~0.02.
    assert by_id["f0"] > 0.5 and by_id["p0"] < 0.1


def test_signal_feeds_stats_layer_and_separates_groups():
    from evalrx.analysis.stats_tools import build_stats_input, run_stats_tool

    batch = _batch()
    result, _ = WhiteboxProbeGenerator(judge=ScriptedJudge(_STRUCT_SINK_PROBE)).generate(
        "attention sink", AttnVLM(), batch, name="struct_sink")
    inp = build_stats_input({"generated_wb:struct_sink": result}, batch)
    key = "generated_wb:struct_sink.struct_sink_mass"
    assert key in inp.per_case
    r = run_stats_tool("signal_label_assoc", inp, {"signal": key})
    assert r.ok and r.effect == 1.0 and r.reject is True  # sink perfectly tracks FAIL


def test_run_cached_recaptures_without_llm():
    gen = WhiteboxProbeGenerator(judge=ScriptedJudge(_STRUCT_SINK_PROBE))
    _, probe = gen.generate("sink", AttnVLM(), _batch(), name="s")
    again = gen.run_cached(probe, AttnVLM(), _batch())
    assert again is not None and again.findings["per_case"]


def test_missing_marker_returns_none():
    gen = WhiteboxProbeGenerator(judge=ScriptedJudge(_NO_MARKER))
    result, probe = gen.generate("x", AttnVLM(), _batch(), name="bad")
    assert result is None and probe is None


def test_non_whitebox_model_yields_nothing():
    gen = WhiteboxProbeGenerator(judge=ScriptedJudge(_STRUCT_SINK_PROBE))
    result, probe = gen.generate("attention", NoAttnModel(), _batch(), name="x")
    assert result is None and probe is None


def test_unavailable_without_backend():
    gen = WhiteboxProbeGenerator()
    assert gen.available is False
    assert gen.generate("x", AttnVLM(), _batch()) == (None, None)


def test_probe_agent_routes_internal_need_to_whitebox():
    """ProbeAgent dispatches an internals-flavoured need to the white-box generator."""
    from evalrx.eval_agent import ProbeAgent

    gen = WhiteboxProbeGenerator(judge=ScriptedJudge(_STRUCT_SINK_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, whitebox_generator=gen)
    need = "measure attention sink mass on structural tokens"
    # An attention-flavoured need on a white-box model routes to the wb generator.
    assert agent._dispatch_generator(need, AttnVLM()) is gen
    # A black-box, output-level need does not.
    assert agent._dispatch_generator("the model refuses to answer", AttnVLM()) is not gen

    # _maybe_generate is the integration seam (probe() resets need each call).
    agent._last_need_custom = need
    results: dict = {}
    added = agent._maybe_generate(AttnVLM(), _batch(), results)
    assert any(a.startswith("generated_wb:") for a in added)
    assert any(k.startswith("generated_wb:") for k in results)
