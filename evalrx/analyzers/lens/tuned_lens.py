"""Tuned lens (Belrose et al.) — per-layer learned affine translators before the unembed.

The logit lens reads layer ``l``'s residual stream as if it were the last
layer's, which is biased: intermediate layers use the residual basis
differently. The tuned lens learns, per layer, an affine translator
``T_l(h) = A_l h + b_l`` (initialised to the identity) such that
``unembed(norm(T_l(h_l)))`` matches the model's own FINAL distribution, by
minimising ``KL(p_final || p_l)``. The fit is label-free: it uses only the
model's outputs, never PASS/FAIL.

Here the translators are fit on token positions sampled from the batch being
analysed (a pretraining corpus is not at hand), or from a fixed set of
``fit_cases``, then each case's final prompt position is read through them;
``fit_kl`` in the findings reports how well the lens fits. With ``fit_cases``
a case's row does not depend on the rest of its batch, so the lens can run on
a growing sample in chunks. Fitted translators are cached on the model per
configuration (and fit set) and can be saved to / loaded from
``translators_path``.

Per-case columns mirror ``logit_lens`` (decision depth, interim peak, late drop)
plus the tuned KL to the final distribution in the middle of the stack.

References:
- Eliciting Latent Predictions from Transformers with the Tuned Lens
  Belrose et al., 2023 — arXiv:2303.08112
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


def _functional_norm(norm: Any, sample: Any) -> Any:
    """A differentiable copy of the final norm with its weights detached.

    Training a translator through the module itself would also write ``.grad``
    into the model's own norm weights; this applies the same function from
    detached tensors instead.
    """
    import torch

    if norm is None:
        return lambda x: x
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)) or 1e-6)
    name = type(norm).__name__.lower()
    if isinstance(norm, torch.nn.LayerNorm) or ("layernorm" in name and "rms" not in name):
        weight = norm.weight.detach().float() if getattr(norm, "weight", None) is not None else None
        bias = norm.bias.detach().float() if getattr(norm, "bias", None) is not None else None

        def layer_norm(x: Any) -> Any:
            return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)

        return layer_norm
    with torch.no_grad():
        w_eff = (norm(torch.ones_like(sample)).float() * (1.0 + eps) ** 0.5).reshape(-1).to(sample.device)

    def rms_norm(x: Any) -> Any:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w_eff

    return rms_norm


@register_analyzer("tuned_lens")
class TunedLensAnalyzer(Analyzer):
    """Tuned-lens readout of the final prompt position, per case.

    Hyper-parameters:
        train_tokens: token positions sampled (across the fit cases) to fit the
                      translators; the final position of every fit case is included.
        fit_cases:    fixed cases the translators are fit on; ``None`` = the
                      analysed batch (then every row depends on the whole batch).
        steps:        Adam steps per layer translator.
        batch_size:   positions per step.
        lr:           Adam learning rate.
        seed:         sampling / shuffling seed.
        translators_path: optional ``.pt`` file to load fitted translators from
                      (when it exists and matches) or save them to.
        max_cases:    label-stratified cap; 0 (the default) = every case.
                      Cost: one hidden-states forward per case, plus the fit.
    """

    name = "tuned_lens"
    description = (
        "Reads each layer's prediction through per-layer translators fit to the model's "
        "own final output (tuned lens); reports how early and how stably the answer forms."
    )
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})
    signal_docs = {
        "tl_decision_frac": ("Answer settles (tuned)", "Relative depth from which every tuned-lens layer already predicts the final answer token; late = the answer forms only at the top."),
        "tl_final_top1_prob": ("Final answer prob.", "Probability of the emitted token at the final layer."),
        "tl_interim_peak_prob": ("Peak answer prob. (tuned)", "Highest tuned-lens probability of the final answer token at any layer."),
        "tl_late_drop": ("Late suppression (tuned)", "Drop from that peak to the final layer; large = the answer was stronger mid-stack and got suppressed."),
        "tl_kl_mid": ("Mid-stack disagreement", "Tuned-lens KL divergence from the final distribution, averaged over the middle third of the layers."),
    }

    def __init__(self, train_tokens: int = 4096, steps: int = 200, batch_size: int = 256,
                 lr: float = 1e-3, seed: int = 0, translators_path: "str | None" = None,
                 fit_cases: "CaseBatch | None" = None, max_cases: int = 0) -> None:
        super().__init__(train_tokens=train_tokens, steps=steps, batch_size=batch_size, lr=lr,
                         seed=seed, translators_path=translators_path, fit_cases=fit_cases,
                         max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        W = model.unembed_weight()
        if W is None:
            raise ValueError(f"{type(model).__name__} exposes no unembed_weight(); the tuned lens "
                             "needs it (white-box local backend).")
        W = W.detach().float()
        device = W.device
        norm = model.final_norm() if hasattr(model, "final_norm") else None

        # One forward per case: keep every layer's final-position state for the
        # readout, plus a seeded sample of positions for fitting the translators
        # (from the fit cases instead, when they are given and not fit yet).
        rows = cases.stratified_head(self.max_cases)
        if not rows:
            raise ValueError("tuned_lens: empty case batch")
        fit_rows = list(self.fit_cases) if self.fit_cases is not None else rows
        if not fit_rows:
            raise ValueError("tuned_lens: empty fit_cases")
        fit_key = None if self.fit_cases is None else (
            "fit_cases", tuple(str(c.id) for c in fit_rows), self.train_tokens, self.steps, self.batch_size,
            self.lr, self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        per_case_budget = max(1, self.train_tokens // len(fit_rows))
        train: "list[list] | None" = None

        def sample(hidden: Any) -> None:
            nonlocal train
            seq = hidden[0].shape[0]
            pick = torch.randperm(seq, generator=gen)[: per_case_budget - 1].tolist()
            pick = sorted(set(pick) | {seq - 1})
            if train is None:
                train = [[] for _ in hidden]
            for layer, h in enumerate(hidden):
                train[layer].append(h[pick].detach().float().cpu())

        last_states: list[tuple[str, list]] = []
        for case in rows:
            hidden = model.forward(case.inputs, capture={Capability.HIDDEN_STATES}).require(
                Capability.HIDDEN_STATES)
            if self.fit_cases is None:
                sample(hidden)
            last_states.append((case.id, [h[-1].detach().float().cpu() for h in hidden]))
        n_layers = len(last_states[0][1]) - 1   # entries 0..L-1 premature, L = final (post-norm)
        dim = last_states[0][1][0].shape[0]
        if self.fit_cases is not None and self._cached(model, fit_key) is None:
            for case in fit_rows:
                sample(model.forward(case.inputs, capture={Capability.HIDDEN_STATES}).require(
                    Capability.HIDDEN_STATES))
        data = [torch.cat(chunks) for chunks in train] if train is not None else None
        apply_norm = _functional_norm(norm, last_states[0][1][0][None].to(device))

        translators, fit_kl = self._translators(model, data, W, apply_norm, n_layers, dim, device, key=fit_key)

        def layer_logprobs(states: list) -> list:
            out = []
            with torch.no_grad():
                for layer in range(n_layers):
                    a, b = translators[layer]
                    x = states[layer].to(device) @ a.T + b
                    out.append(torch.log_softmax(apply_norm(x) @ W.T, dim=-1))
                out.append(torch.log_softmax(states[n_layers].to(device) @ W.T, dim=-1))
            return out

        mid = list(range(n_layers // 3, max(n_layers // 3 + 1, 2 * n_layers // 3)))
        per_case: list[dict[str, Any]] = []
        for case_id, states in last_states:
            logps = layer_logprobs(states)
            final = logps[-1]
            top1 = int(final.argmax())
            probs = [float(lp[top1].exp()) for lp in logps]
            argmaxes = [int(lp.argmax()) for lp in logps]
            decision = next((i for i in range(len(logps)) if all(a == top1 for a in argmaxes[i:])),
                            len(logps) - 1)
            kls = [float(torch.sum(final.exp() * (final - logps[i]))) for i in mid]
            peak = max(probs)
            per_case.append({
                "sample_id": case_id,
                "tl_decision_frac": round(decision / max(1, len(logps) - 1), 4),
                "tl_final_top1_prob": round(probs[-1], 4),
                "tl_interim_peak_prob": round(peak, 4),
                "tl_late_drop": round(peak - probs[-1], 4),
                "tl_kl_mid": round(sum(kls) / len(kls), 4),
            })

        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"fit_kl_per_layer": fit_kl},
            findings={
                "n_cases_analyzed": len(per_case),
                "n_layers": n_layers,
                "n_train_tokens": int(data[0].shape[0]) if data is not None else None,
                "fit_on": "fit_cases" if self.fit_cases is not None else "batch",
                "fit_kl_per_layer": [round(k, 4) for k in fit_kl],
                "fit_kl_mean": round(sum(fit_kl) / max(1, len(fit_kl)), 4),
                "final_norm_applied": norm is not None,
                "per_case": per_case,
                "_caveat": (
                    "Translators are fit label-free on token positions of "
                    + ("the fixed fit_cases" if self.fit_cases is not None else
                       "the analysed batch itself (no separate corpus), so they are in-sample for these prompts")
                    + "; fit_kl_per_layer is the held-in KL after fitting."
                ),
            },
        )

    def _cache(self, model: Any) -> dict:
        try:
            from evalrx.analyzers._internals.access import torch_module

            module = torch_module(model, self.name)
        except ValueError:
            return {}
        return module.__dict__.setdefault("_evalrx_tuned_lens", {})

    def _cached(self, model: Any, key: Any) -> "tuple[list, list[float]] | None":
        return self._cache(model).get(key)

    def _translators(self, model: Any, data: "list | None", W: Any, apply_norm: Any, n_layers: int,
                     dim: int, device: Any, key: Any = None) -> "tuple[list, list[float]]":
        """Fit (or load / reuse) one affine translator per premature layer."""
        import os

        import torch

        if key is None:
            key = (n_layers, dim, int(data[0].shape[0]), self.steps, self.batch_size, self.lr, self.seed)
        cache = self._cache(model)
        if key in cache:
            return cache[key]
        path = self.translators_path
        if path and os.path.exists(path):
            saved = torch.load(path, map_location=device)
            if saved.get("n_layers") == n_layers and saved.get("dim") == dim:
                fitted = ([(a.to(device), b.to(device)) for a, b in saved["translators"]],
                          list(saved["fit_kl"]))
                cache[key] = fitted
                return fitted

        gen = torch.Generator().manual_seed(self.seed)
        final = data[n_layers]
        translators, fit_kl = [], []
        for layer in range(n_layers):
            x_all = data[layer]
            a = torch.eye(dim, device=device, requires_grad=True)
            b = torch.zeros(dim, device=device, requires_grad=True)
            opt = torch.optim.Adam([a, b], lr=self.lr)
            for _ in range(self.steps):
                idx = torch.randint(0, x_all.shape[0], (min(self.batch_size, x_all.shape[0]),), generator=gen)
                x, target = x_all[idx].to(device), final[idx].to(device)
                with torch.no_grad():
                    log_p = torch.log_softmax(target @ W.T, dim=-1)
                log_q = torch.log_softmax(apply_norm(x @ a.T + b) @ W.T, dim=-1)
                loss = torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                kl = 0.0
                for start in range(0, x_all.shape[0], 1024):
                    x, target = x_all[start:start + 1024].to(device), final[start:start + 1024].to(device)
                    log_p = torch.log_softmax(target @ W.T, dim=-1)
                    log_q = torch.log_softmax(apply_norm(x @ a.T + b) @ W.T, dim=-1)
                    kl += float(torch.nn.functional.kl_div(log_q, log_p, log_target=True, reduction="sum"))
            translators.append((a.detach(), b.detach()))
            fit_kl.append(kl / x_all.shape[0])
        fitted = (translators, fit_kl)
        cache[key] = fitted
        if path:
            torch.save({"n_layers": n_layers, "dim": dim, "fit_kl": fit_kl,
                        "translators": [(a.cpu(), b.cpu()) for a, b in translators]}, path)
        return fitted
