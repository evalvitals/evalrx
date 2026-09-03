"""Unit tests for TCD's pure math (evalrx.models.paper_methods.tcd).

No GPU/model weights needed here -- these exercise the Eq. 1-9 formulas
directly against hand-built tensors. End-to-end wiring (encoder hooking,
prefill attention capture, the two-KV-cache decode loop in
``HFLocalModel.generate_tcd``) was verified separately against real
Qwen2-Audio-7B-Instruct weights on GPU; see the commit message.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evalrx.core.spec import AudioSpec, ModelSpec
from evalrx.models.backends.base import RuntimeConfig
from evalrx.models.backends.hf_local import HFLocalModel
from evalrx.models.paper_methods import tcd


def test_paper_method_fidelity_tcd():
    audio_spec = ModelSpec(
        key="qwen2-audio-7b-instruct", family="fake", model_type="fake_audio", hf_repo="",
        audio=AudioSpec(audio_token_id_attr="audio_token_id", audio_tower="audio_tower"),
    )
    assert HFLocalModel(audio_spec, RuntimeConfig()).paper_method_fidelity("tcd") == (
        "native_layer_matched_stability"
    )

    mismatched_spec = ModelSpec(
        key="fake-omni", family="fake", model_type="fake_omni", hf_repo="",
        audio=AudioSpec(audio_token_id_attr="audio_token_id", audio_tower="audio_tower"),
    )
    assert HFLocalModel(mismatched_spec, RuntimeConfig()).paper_method_fidelity("tcd") == (
        "adapted_truncated_layer_stability"
    )

    text_only_spec = ModelSpec(key="fake-llm", family="fake", model_type="fake_llm", hf_repo="")
    assert HFLocalModel(text_only_spec, RuntimeConfig()).paper_method_fidelity("tcd") == "unavailable"


def test_hann_blur_waveform_preserves_length_and_rms():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000).astype(np.float32) * 0.1
    blurred = tcd.hann_blur_waveform(wav, 16000, window_ms=20.0)
    assert blurred.shape == wav.shape
    assert blurred.dtype == np.float32
    # Eq. 1's "rescaling x_tilde to preserve global amplitude": RMS should
    # match the original to a tight tolerance, even though the waveform
    # itself is now smoother (blurring alone would shrink RMS).
    orig_rms = np.sqrt(np.mean(wav**2))
    blur_rms = np.sqrt(np.mean(blurred**2))
    assert blur_rms == pytest.approx(orig_rms, rel=1e-3)


def test_hann_blur_waveform_smooths_high_frequency_content():
    sr = 16000
    t = np.arange(sr) / sr
    wav = np.sin(2 * np.pi * 3000 * t).astype(np.float32)  # a fast-varying tone
    blurred = tcd.hann_blur_waveform(wav, sr, window_ms=20.0)
    # A wide blur window on a high-frequency tone should reduce sample-to-sample
    # flux relative to the original -- this is the whole point of Eq. 1's
    # "slow-path view".
    orig_flux = np.mean(np.abs(np.diff(wav)))
    blur_flux = np.mean(np.abs(np.diff(blurred)))
    assert blur_flux < orig_flux


def test_hann_blur_waveform_degenerate_window_is_near_identity():
    wav = np.array([1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32)
    blurred = tcd.hann_blur_waveform(wav, sample_rate=16000, window_ms=0.0)
    assert blurred.shape == wav.shape


def test_layer_stability_bounded_and_monotone_in_flux():
    torch.manual_seed(0)
    stable = [torch.ones(10, 4) * (i + 1) for i in range(3)]  # constant per layer -> zero flux
    scores = tcd.layer_stability(stable, eps=1e-6)
    assert scores.shape == (3,)
    assert torch.all(scores > 0.99)  # near-zero flux -> stability near 1

    noisy = [torch.randn(10, 4) * 100 for _ in range(3)]  # large frame-to-frame jumps
    noisy_scores = tcd.layer_stability(noisy, eps=1e-6)
    assert torch.all((noisy_scores >= 0) & (noisy_scores <= 1))
    assert float(noisy_scores.mean()) < float(scores.mean())


def test_audio_attention_ratio_isolates_masked_positions():
    # 1 head, 1 query row, 4 keys; all attention mass on keys [1, 3]
    attn = torch.tensor([[[0.0, 0.5, 0.0, 0.5]]])
    mask = torch.tensor([False, True, False, True])
    assert float(tcd.audio_attention_ratio(attn, mask)) == pytest.approx(1.0)

    mask_none = torch.tensor([True, False, True, False])
    assert float(tcd.audio_attention_ratio(attn, mask_none)) == pytest.approx(0.0)


def test_aggregate_stability_favors_high_attention_layers():
    stability = torch.tensor([0.1, 0.9])
    # layer 1 gets essentially all the softmax weight at a sharp temperature
    ratio_favoring_layer1 = torch.tensor([0.0, 1.0])
    S = tcd.aggregate_stability(stability, ratio_favoring_layer1, temperature=50.0)
    assert S == pytest.approx(0.9, abs=1e-3)


def test_aggregate_stability_requires_equal_length_inputs():
    with pytest.raises(ValueError, match="equal-length"):
        tcd.aggregate_stability(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5]), temperature=4.0)


def test_adaptive_blur_params_respects_table6_bounds():
    hp = tcd.TCDHyperparams()
    w0, lam0 = tcd.adaptive_blur_params(0.0, hp)
    w1, lam1 = tcd.adaptive_blur_params(1.0, hp)
    assert w0 == pytest.approx(hp.w_min)
    assert w1 == pytest.approx(hp.w_max)
    assert lam0 == pytest.approx(hp.lam_min)
    assert lam1 == pytest.approx(hp.lam_max)


def test_topk_renormalized_entropy_bounds():
    # A one-hot distribution has zero entropy even after renormalizing top-K.
    peaked = torch.full((100,), -10.0)
    peaked[0] = 10.0
    assert tcd.topk_renormalized_entropy(peaked, k=5) == pytest.approx(0.0, abs=1e-4)

    # A uniform distribution's top-K is itself uniform -> max normalized entropy.
    uniform = torch.zeros(100)
    assert tcd.topk_renormalized_entropy(uniform, k=5) == pytest.approx(1.0, abs=1e-4)


def test_reliance_gate_is_capped_at_one_and_zero_when_reliance_is_zero():
    hp = tcd.TCDHyperparams()
    assert tcd.reliance_gate(0.0, 1.0, hp) == 0.0
    assert tcd.reliance_gate(10.0, 1.0, hp) == pytest.approx(1.0)  # clamps at Eq. 8's 1.0 cap


def test_fuse_logits_only_touches_candidate_set_and_never_masks():
    hp = tcd.TCDHyperparams(k_orig=2, k_blur=2)
    z = torch.tensor([1.0, 5.0, 2.0, 0.0])
    z_tilde = torch.tensor([1.0, 0.0, 2.0, 3.0])
    fused = tcd.fuse_logits(z, z_tilde, lam=1.0, gate_value=1.0, hp=hp)

    # top-2 of z: indices {1, 2}; top-2 of z_tilde: indices {3, 2} -> Omega = {1, 2, 3}
    # index 0 is outside Omega_t: TCD applies NO cutoff masking (unlike VCD/IFCD/PAI),
    # so it must come back byte-identical to the original logit.
    assert fused[0] == pytest.approx(z[0])
    # index 1: in original top-K only, z_tilde < z there -> positive update applies
    assert fused[1] == pytest.approx(z[1] + (z[1] - z_tilde[1]))
    # index 3: z_tilde > z there -> ReLU(z - z_tilde) == 0, so no change despite being a candidate
    assert fused[3] == pytest.approx(z[3])


def test_fuse_logits_gate_zero_is_a_no_op():
    hp = tcd.TCDHyperparams()
    z = torch.randn(50)
    z_tilde = torch.randn(50)
    fused = tcd.fuse_logits(z, z_tilde, lam=1.5, gate_value=0.0, hp=hp)
    assert torch.allclose(fused, z)


def test_hyperparams_match_appendix_a_table_6():
    # Values read directly off the PDF's Table 6 (Appendix A), not re-derived.
    hp = tcd.TCDHyperparams()
    assert hp.l_attn == 4
    assert hp.tau == pytest.approx(4.0)
    assert (hp.w_min, hp.w_max) == pytest.approx((8.0, 30.0))
    assert (hp.lam_min, hp.lam_max) == pytest.approx((0.3, 1.5))
    assert hp.k_orig == 16
    assert hp.k_blur == 8
    assert hp.gamma_gate == pytest.approx(2.0)  # Qwen2-Audio-Instruct value, Appendix A.2(4)
    assert hp.alpha_entropy == pytest.approx(0.5)
    assert hp.k_ent == 5
    assert hp.eps == pytest.approx(1e-6)
