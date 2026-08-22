"""Linear probing — per-layer representation richness for a target property.

Train a linear classifier on each layer's hidden states (samples + labels) and
report per-layer HELD-OUT accuracy — where in the stack is the property linearly
decodable.  Labels come from ``case.label`` (PASS/FAIL), so the probe answers:
"at which depth do the representations already separate failing from passing
inputs".  Correlational, not causal.

Implementation notes (no sklearn dependency — torch only):
- one forward per case (``HIDDEN_STATES``), feature = hidden state at ``pos``;
- per-layer logistic regression, class-weighted BCE + L2, Adam;
- stratified 2-fold cross-validation: every reported number and every
  ``per_case`` probability is out-of-fold (never train-set);
- balanced accuracy (mean of per-class recalls) — robust to FAIL/PASS skew;
- too few labelled cases per class -> returns a SKIPPED result with the reason
  instead of raising (an analyzer crash silently removes the evidence slot
  from the loop's view; a degraded honest result does not).

References:
- Understanding intermediate layers using linear classifier probes
  Alain & Bengio, 2016 — arXiv:1610.01644
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@register_analyzer("linear_probe")
class LinearProbeAnalyzer(Analyzer):
    """Per-layer linear-probe (held-out, balanced) accuracy for PASS/FAIL."""

    name = "linear_probe"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    # max_cases sized for enriched batches (see LogitLensAnalyzer): the held-out
    # probe needs enough of the scarce FAIL class for a trustworthy accuracy
    # curve and a powered downstream contrast; one forward per case.
    def __init__(self, pos: int = -1, max_cases: int = 0, min_per_class: int = 4,
                 l2: float = 1e-2, epochs: int = 200, seed: int = 0) -> None:
        super().__init__(pos=pos, max_cases=max_cases, min_per_class=min_per_class,
                         l2=l2, epochs=epochs, seed=seed)

    # ------------------------------------------------------------------
    def _fit_predict(self, X_tr, y_tr, X_te):
        """Class-weighted logistic regression; returns P(fail) on X_te."""
        import torch

        torch.manual_seed(self.seed)
        d = X_tr.shape[1]
        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        n_pos = float(y_tr.sum())
        pos_weight = torch.tensor((len(y_tr) - n_pos) / max(n_pos, 1.0))
        opt = torch.optim.Adam([w, b], lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(X_tr @ w + b, y_tr) + self.l2 * w.pow(2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return torch.sigmoid(X_te @ w + b)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalvitals.core.case import CaseBatch, Label

        # Label-stratified subsample so the capped probe pool keeps enough FAIL
        # cases (a plain head is mostly PASS on an enriched batch).
        labelled_batch = CaseBatch([c for c in cases
                                    if c.label in (Label.PASS, Label.FAIL)])
        labelled = labelled_batch.stratified_head(self.max_cases)
        n_fail = sum(1 for c in labelled if c.label == Label.FAIL)
        n_pass = len(labelled) - n_fail

        def skipped(reason: str) -> Result:
            return Result(
                analyzer=self.name, model=repr(model), cases=cases,
                findings={"skipped": reason, "n_fail": n_fail, "n_pass": n_pass,
                          "n_layers": None, "per_layer_accuracy": [],
                          "best_layer": None, "best_accuracy": None, "per_case": []},
            )

        if min(n_fail, n_pass) < self.min_per_class:
            return skipped(f"needs >= {self.min_per_class} cases per class "
                           f"(got fail={n_fail}, pass={n_pass})")

        # -- collect features: one forward per case, all layers at `pos` ----
        feats, ys, ids = [], [], []
        for case in labelled:
            trace = model.forward(case.inputs, capture={Capability.HIDDEN_STATES})
            hidden = trace.require(Capability.HIDDEN_STATES)
            feats.append(torch.stack([h[self.pos].detach().float().cpu() for h in hidden]))
            ys.append(1.0 if case.label == Label.FAIL else 0.0)
            ids.append(case.id)
        X = torch.stack(feats)            # (n, L, d)
        y = torch.tensor(ys)
        n, n_layers, _ = X.shape

        # -- stratified 2-fold assignment (seeded, deterministic) -----------
        g = torch.Generator().manual_seed(self.seed)
        fold = torch.zeros(n, dtype=torch.long)
        for cls in (0.0, 1.0):
            idx = [i for i in range(n) if ys[i] == cls]
            perm = torch.randperm(len(idx), generator=g)
            for j, p in enumerate(perm):
                fold[idx[p]] = j % 2

        per_layer_acc, oof = [], torch.zeros(n_layers, n)
        for layer in range(n_layers):
            Xl = X[:, layer, :]
            accs = []
            for f in (0, 1):
                tr, te = (fold != f), (fold == f)
                mu, sd = Xl[tr].mean(0), Xl[tr].std(0).clamp_min(1e-6)
                p = self._fit_predict((Xl[tr] - mu) / sd, y[tr], (Xl[te] - mu) / sd)
                oof[layer, te] = p
                pred = (p > 0.5).float()
                yt = y[te]
                rec_f = ((pred == 1) & (yt == 1)).sum() / (yt == 1).sum().clamp_min(1)
                rec_p = ((pred == 0) & (yt == 0)).sum() / (yt == 0).sum().clamp_min(1)
                accs.append(float((rec_f + rec_p) / 2))
            per_layer_acc.append(round(sum(accs) / len(accs), 4))

        best_layer = max(range(n_layers), key=lambda i: per_layer_acc[i])
        per_case = [{
            "sample_id": ids[i],
            "fail_prob_best_layer": round(float(oof[best_layer, i]), 4),
            "is_fail": bool(ys[i]),
        } for i in range(n)]

        return Result(
            analyzer=self.name, model=repr(model), cases=cases,
            artifacts={"per_layer_accuracy": per_layer_acc, "per_case": per_case},
            findings={
                "n_layers": n_layers, "pos": self.pos,
                "n_cases": n, "n_fail": n_fail, "n_pass": n_pass,
                "per_layer_accuracy": per_layer_acc,
                "best_layer": best_layer,
                "best_layer_frac": round(best_layer / max(1, n_layers - 1), 4),
                "best_accuracy": per_layer_acc[best_layer],
                "chance": 0.5,  # balanced accuracy baseline
                "per_case": per_case,
            },
        )
