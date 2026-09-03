"""Temporal Contrastive Decoding (TCD) — Li et al. 2026, arXiv:2604.15383.

TCD is NOT a VCD-family port. VCD/IFCD/PAI all share one combine rule
(``(1+alpha)*z - alpha*z_tilde`` masked below a ``max + log(beta)`` cutoff,
see :mod:`evalrx.models.paper_methods.vcd`); TCD's is additive,
positive-rectified, and restricted to a small candidate set, with no cutoff
masking at all (Eq. 7-9 below). The two other structural differences from
VCD's per-step-noisy-forward pattern:

* the "corrupted" view is a **waveform** operation (Hann-window blur, Eq. 1)
  followed by a full audio-encoder re-encode, computed ONCE before decoding
  starts — not a per-step latent perturbation;
* the blur window and update scale are **per-example adaptive**, derived from
  a self-normalized stability score (Eq. 2-6) that needs the audio encoder's
  per-layer hidden-state trajectory AND the decoder's per-layer attention to
  audio tokens — signals VCD/IFCD/PAI never touch.

This module holds pure, framework-light math only (all Eq. references are to
the paper's Section 3 / Appendix A). The orchestration — encoding, KV-cache
management, the two-branch decode loop — lives in
``HFLocalModel.generate_tcd`` (``models/backends/hf_local.py``), matching
this package's split of "paper math here, model plumbing in the backend".

Hyperparameter defaults (:class:`TCDHyperparams`) are Table 6 verbatim, read
directly from the PDF (Appendix A) — not re-derived, not taken from a
WebFetch summary of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TCDHyperparams:
    """Table 6 defaults. The paper uses ONE configuration across benchmarks;
    only ``gamma_gate`` is set per backbone (2.0 for Qwen2-Audio-Instruct,
    the paper's own hyperparameter anchor — see Appendix A.1/A.2(4))."""

    l_attn: int = 4          # last decoder layers aggregated for the step-wise gate's r_t (Eq. 8)
    tau: float = 4.0          # stability aggregation softmax temperature (Eq. 4)
    w_min: float = 8.0        # blur window bounds, ms (Eq. 5)
    w_max: float = 30.0
    lam_min: float = 0.3      # update-scale bounds (Eq. 6)
    lam_max: float = 1.5
    k_orig: int = 16          # candidate set from the original logits (Eq. 9's Omega_t)
    k_blur: int = 8           # candidate set from the slow-path logits
    gamma_gate: float = 2.0   # gate multiplier; paper value for Qwen2-Audio-Instruct
    alpha_entropy: float = 0.5   # entropy exponent in the gate (Eq. 8)
    k_ent: int = 5            # top-K for the normalized entropy signal (Eq. 8)
    eps: float = 1e-6         # numerical stability term (Eq. 3)


def hann_blur_waveform(waveform: Any, sample_rate: int, window_ms: float) -> Any:
    """Eq. 1: temporally blur the waveform with a normalized Hann window, then
    rescale to preserve global RMS amplitude ("rescaling x_tilde to preserve
    global amplitude", Section 3.2).

    ``waveform`` is a 1-D mono float32 numpy array (matches
    ``hf_local._resolve_audio``'s contract). Returns the blurred waveform at
    the same length and sample rate — it is re-encoded through the SAME
    processor path as the original audio, not treated specially downstream.
    """
    import numpy as np

    win_len = max(1, int(round(float(window_ms) / 1000.0 * sample_rate)))
    if win_len % 2 == 0:
        win_len += 1  # odd length -> symmetric window, unambiguous center tap
    if win_len <= 1:
        return np.asarray(waveform, dtype=np.float32).copy()

    window = np.hanning(win_len).astype(np.float32)
    window_sum = window.sum()
    if window_sum <= 0:  # pragma: no cover - hanning(n>=3) is always positive-summed
        return np.asarray(waveform, dtype=np.float32).copy()
    window = window / window_sum

    pad = win_len // 2
    padded = np.pad(np.asarray(waveform, dtype=np.float32), (pad, pad), mode="reflect")
    blurred = np.convolve(padded, window, mode="valid").astype(np.float32)

    orig_rms = float(np.sqrt(np.mean(np.square(waveform)))) + 1e-8
    blur_rms = float(np.sqrt(np.mean(np.square(blurred)))) + 1e-8
    return (blurred * (orig_rms / blur_rms)).astype(np.float32)


def layer_stability(hidden_states: list, *, eps: float) -> Any:
    """Eq. 2-3: per-layer magnitude ``M``, temporal flux ``F``, and stability
    ``S_l = M_l / (M_l + F_l + eps)``.

    ``hidden_states`` is a list of per-layer encoder hidden-state tensors,
    each ``(seq_len, dim)`` with the batch dimension already stripped (one
    entry per encoder layer, time-ordered along ``seq_len`` — the audio
    encoder's own frame axis, NOT decoding steps). Returns a 1-D tensor of
    per-layer stability scores, same length as ``hidden_states``.
    """
    import torch

    magnitudes, fluxes = [], []
    for h in hidden_states:
        h = h.float()
        magnitudes.append(h.norm(dim=-1).mean())
        if h.shape[0] > 1:
            fluxes.append((h[1:] - h[:-1]).norm(dim=-1).mean())
        else:  # pragma: no cover - a single-frame encoder trajectory is degenerate input
            fluxes.append(torch.zeros((), dtype=h.dtype, device=h.device))
    M = torch.stack(magnitudes)
    F = torch.stack(fluxes)
    return M / (M + F + eps)


def audio_attention_ratio(attn_layer: Any, audio_mask: Any) -> Any:
    """Fraction of one decoder layer's attention mass landing on audio-token
    key positions, averaged over heads and query rows.

    ``attn_layer`` is ``(heads, q, k)`` (batch stripped); ``audio_mask`` is a
    bool tensor ``(k,)``. Shared by both Eq. 4's per-layer ``r_l`` (all
    layers, computed once from the prefill) and Eq. 8's step-wise ``r_t``
    (last ``l_attn`` layers, recomputed at every decode step) — same
    quantity, different layer subset and call frequency.
    """
    attn_layer = attn_layer.float()
    audio_mass = attn_layer[..., audio_mask].sum(dim=-1)
    total_mass = attn_layer.sum(dim=-1).clamp_min(1e-8)
    return (audio_mass / total_mass).mean()


def aggregate_stability(stability_per_layer: Any, attention_ratio_per_layer: Any, *, temperature: float) -> float:
    """Eq. 4: softmax-weight per-layer stability by audio-attention ratio.

    Both inputs are 1-D tensors of equal length. This equal-length
    requirement is exactly the point in the paper where per-layer stability
    (an ENCODER-side quantity) and per-layer attention ratio (a DECODER-side
    quantity) get zipped together index-for-index — faithful only when the
    audio encoder and text decoder have the same layer count (true for
    Qwen2-Audio-Instruct: 32 encoder / 32 decoder layers; NOT true for e.g.
    Qwen2.5-Omni's 32/28 split). Callers must truncate both inputs to
    ``min(len(...), len(...))`` themselves when depths differ, and report
    ``paper_method_fidelity("tcd")`` as ``"adapted"`` in that case.
    """
    import torch

    if stability_per_layer.shape[0] != attention_ratio_per_layer.shape[0]:
        raise ValueError(
            "aggregate_stability requires equal-length per-layer inputs "
            f"(got {stability_per_layer.shape[0]} stability vs "
            f"{attention_ratio_per_layer.shape[0]} attention-ratio entries); "
            "truncate to min(...) before calling for a mismatched-depth architecture"
        )
    weights = torch.softmax(temperature * attention_ratio_per_layer.float(), dim=0)
    return float((weights * stability_per_layer.float()).sum())


def adaptive_blur_params(stability: float, hp: TCDHyperparams) -> "tuple[float, float]":
    """Eq. 5-6: map the scalar stability score to (blur window ms, update scale)."""
    window_ms = hp.w_min + (hp.w_max - hp.w_min) * stability
    lam = hp.lam_min + (hp.lam_max - hp.lam_min) * stability
    return float(window_ms), float(lam)


def topk_renormalized_entropy(logits: Any, k: int) -> float:
    """Eq. 8's uncertainty signal: normalized entropy of softmax(logits),
    RENORMALIZED over its own top-K probability mass first, then divided by
    log(K) so the result is bounded in [0, 1]."""
    import torch

    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, min(int(k), probs.shape[-1]))
    top_probs = top.values / top.values.sum().clamp_min(1e-12)
    entropy = -(top_probs * top_probs.clamp_min(1e-12).log()).sum()
    denom = math.log(top.values.shape[-1]) if top.values.shape[-1] > 1 else 1.0
    return float(entropy / denom)


def reliance_gate(audio_reliance: float, entropy_hat: float, hp: TCDHyperparams) -> float:
    """Eq. 8: ``g_t = min(gamma_gate * r_t * H_hat_t ** alpha, 1.0)``."""
    return min(hp.gamma_gate * float(audio_reliance) * (max(entropy_hat, 0.0) ** hp.alpha_entropy), 1.0)


def fuse_logits(z: Any, z_tilde: Any, *, lam: float, gate_value: float, hp: TCDHyperparams) -> Any:
    """Eq. 7-9: gated, candidate-restricted, positive-only logit update.

    Unlike VCD/IFCD/PAI, there is NO cutoff masking here — tokens outside the
    candidate set ``Omega_t`` are returned UNCHANGED (``z_t(j)`` verbatim),
    they are not suppressed to ``-inf``. Only the union of each branch's
    top-K is nudged, and only upward (``ReLU(z - z_tilde)``, Eq. 7), scaled
    by ``lam * gate_value``.
    """
    import torch

    positive_diff = (z - z_tilde).clamp_min(0.0)
    orig_idx = torch.topk(z, min(hp.k_orig, z.shape[-1])).indices
    blur_idx = torch.topk(z_tilde, min(hp.k_blur, z_tilde.shape[-1])).indices
    candidates = torch.zeros_like(z, dtype=torch.bool)
    candidates[orig_idx] = True
    candidates[blur_idx] = True

    fused = z.clone()
    fused[candidates] = z[candidates] + lam * gate_value * positive_diff[candidates]
    return fused
