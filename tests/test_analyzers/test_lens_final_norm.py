"""Regression: lens analyzers must NOT re-normalise the final hidden-states entry.

HF backends return ``outputs.hidden_states`` whose LAST element is already post
final-norm (Llama/Qwen/GPT2 convention, and both lens analyzers request the
full stack themselves).  Applying ``final_norm`` to it again squares the RMS
gain, shifting the reference distribution every decision-depth / divergence
column is anchored to.  These tests build a model whose norm has a non-unit
gain and pin the corrected behavior quantitatively.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from evalrx.analyzers.lens.layer_contrast import LayerContrastAnalyzer  # noqa: E402
from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer  # noqa: E402
from evalrx.core.capability import Capability  # noqa: E402
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label  # noqa: E402
from evalrx.core.model import Model, Trace  # noqa: E402

_GAIN = 2.0
_DIM = 4


class _Scale(torch.nn.Module):
    """Stand-in for an RMSNorm with learned gain _GAIN on every dim."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((_DIM,), _GAIN))

    def forward(self, x):
        return x * self.weight


class NormedFakeModel(Model):
    """Two-layer stack following the HF convention: last entry is post-norm."""

    capabilities = frozenset({Capability.GENERATE, Capability.HIDDEN_STATES})
    modalities = frozenset({"text"})

    def __init__(self):
        self._norm = _Scale()
        # premature layer favours token 2; the raw final layer favours token 3
        self.h_premature = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        h_final_raw = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.h_final_postnorm = self._norm(h_final_raw).detach()  # [0,0,0,2]

    def generate(self, inputs, **kwargs):  # pragma: no cover
        return ""

    def unembed_weight(self):
        return torch.eye(_DIM)

    def final_norm(self):
        return self._norm

    def forward(self, inputs, capture, spec=None):
        return Trace(
            tokens=["x"],
            token_ids=[0],
            provided={Capability.HIDDEN_STATES},
            hidden_states=[self.h_premature, self.h_final_postnorm],
        )


def _batch():
    return CaseBatch([FailureCase(inputs=Inputs(prompt="q"), label=Label.FAIL)])


def _expected_final_top1_prob():
    # softmax([0,0,0,GAIN])[3] — single norm; the double-norm bug gave GAIN^2
    z = [0.0, 0.0, 0.0, _GAIN]
    exps = [math.exp(v) for v in z]
    return exps[3] / sum(exps)


def test_logit_lens_final_entry_projected_as_is():
    f = LogitLensAnalyzer().run(NormedFakeModel(), _batch()).findings
    entry = f["per_case"][0]
    assert f["final_norm_applied"] is True
    assert entry["final_top1_prob"] == pytest.approx(_expected_final_top1_prob(), abs=1e-3)
    # premature layer IS normed: its top-1 (token 2) differs from final (token 3),
    # so the decision forms only at the final layer
    assert entry["decision_layer"] == 1


def test_layer_contrast_final_entry_projected_as_is():
    f = LayerContrastAnalyzer().run(NormedFakeModel(), _batch()).findings
    entry = f["per_case"][0]
    assert entry["final_top1_prob"] == pytest.approx(_expected_final_top1_prob(), abs=1e-3)
    # premature layer disagrees with final -> zero agreement in the window
    assert entry["layer_agreement_frac"] == 0.0
    assert entry["jsd_max"] > 0.0
