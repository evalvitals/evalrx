"""Weight-edit audit — does the edit change behaviour, and is the change specific?

An edit that moves the model's answers has not yet been explained: a change of
the same size along a random direction, or the same rule on other neurons, may
move them just as much, and a constant shift of the activations may do the same
job without touching the weights. This analyzer measures the edit beside those
alternatives on the same cases and the same scalar.

Per case, at the last prompt position (where the first answer token is
predicted), it records the next-token margin (see
:mod:`evalrx.analyzers._internals.targets`) and the KL divergence from the
unedited distribution under:

* the edit itself;
* ``random_direction`` controls — same neurons and change size, random direction;
* ``random_columns`` controls — same rule, other neurons;
* a steering control — weights untouched, a constant vector added to each
  edited layer's output equal to the edit's mean effect there.

Reading the result:

* ``specificity_*`` near 1 means a matched random change moves the output as
  much as the edit: the direction (or the neurons) is not what matters.
* ``steer_recovery`` near 1 means a constant activation shift reproduces the
  edit's effect on the margin: the edit acts like an input-independent bias.
* with labels, ``mean_delta_edit_pass`` and ``frac_flips_pass`` are the side
  effects on cases the model already got right.

Everything is measured teacher-forced on one position. An edit whose effect
only appears during generation (for example on repetition loops) needs a
generation-level evaluation, which the fix stage provides.

References:
- Sparks of Weight-Space Programming — Takehi, Teoh, Zhang & Khattab, 2026.
- EditLens: What Did Astra's Edit Actually Do? — Zhang, Shang & Li, 2026.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.registry import register_analyzer
from evalrx.core.result import Result

if TYPE_CHECKING:
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model


@register_analyzer("weight_edit_audit")
class WeightEditAudit(Analyzer):
    """Measure a weight edit against random-direction, random-neuron and steering controls.

    Args:
        edit:      The :class:`~evalrx.wsp.edit.WeightEdit` to audit.
        target:    ``"prediction"`` (the clean run's top-1 vs top-2 token; label
                   free) or ``"gold"`` (the gold answer's first token vs the
                   strongest other token; a positive change moves toward gold).
        controls:  Control kinds to run (see :mod:`evalrx.wsp.controls`).
        n_seeds:   Random draws per control kind; their per-case results are
                   averaged.
        steering:  Whether to run the constant-shift steering control.
        seed:      Base seed of the control draws.
        max_cases: Analyse at most this many cases (0 = all), stratified by label.
    """

    name = "weight_edit_audit"
    requires = frozenset({Capability.LOGITS})
    question = "Does the edit change the answer, and more than a matched random change would?"
    signal_docs = {
        "we_delta_edit": (
            "Margin change (edit)",
            "Change in the next-token margin caused by the edit. With the gold target, "
            "positive means the edit moved the model toward the right answer.",
        ),
        "we_kl_edit": (
            "Output shift (edit)",
            "How far the edit moves the next-token distribution from the unedited one "
            "(KL divergence). Zero means the edit did nothing on this case.",
        ),
        "we_flips_top1": (
            "Top answer changed",
            "Whether the edit changed the model's most likely next token on this case.",
        ),
        "we_kl_random_direction": (
            "Output shift (random dir.)",
            "The same measurement for an equally large change along a random direction. "
            "If it is as large as the edit's, the edit's direction is not what matters.",
        ),
        "we_kl_random_columns": (
            "Output shift (other neurons)",
            "The same measurement for the same rule applied to other neurons. If it is as "
            "large as the edit's, the choice of neurons is not what matters.",
        ),
        "we_delta_steer": (
            "Margin change (steering)",
            "Change in the margin when the weights are left alone and a constant vector is "
            "added to the edited layers' outputs instead. Close to the edit's own change "
            "means the edit behaves like a fixed bias.",
        ),
    }

    edit: Any
    target: str
    controls: tuple[str, ...]
    n_seeds: int
    steering: bool
    seed: int
    max_cases: int

    def __init__(self, edit: "Any", target: str = "prediction",
                 controls: "tuple[str, ...]" = ("random_direction", "random_columns"),
                 n_seeds: int = 3, steering: bool = True, seed: int = 0,
                 max_cases: int = 0) -> None:
        super().__init__(edit=edit, target=target, controls=tuple(controls), n_seeds=n_seeds,
                         steering=steering, seed=seed, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.targets import build_margin
        from evalrx.core.case import Label
        from evalrx.core.model import CaptureSpec
        from evalrx.wsp.controls import CONTROL_KINDS, build_control
        from evalrx.wsp.direction import output_shift
        from evalrx.wsp.edit import applied, module_of, resolve_parameters
        from evalrx.wsp.structure import edit_structure

        unknown = [k for k in self.controls if k not in CONTROL_KINDS]
        if unknown:
            raise ValueError(f"{self.name}: unknown control(s) {unknown}; choose from {CONTROL_KINDS}")
        if int(self.n_seeds) < 1:
            raise ValueError(f"{self.name}: n_seeds must be at least 1")
        module = module_of(model)
        resolved = resolve_parameters(module, self.edit)
        spec = CaptureSpec(to_cpu=False)

        def last_logits(inputs: Any) -> Any:
            trace = model.forward(inputs, capture={Capability.LOGITS}, spec=spec)
            return trace.require(Capability.LOGITS)[-1:].detach().float().cpu()

        # Inputs of the edited linear layers at the last position, for the
        # steering control: its shift is the edit's mean effect dW @ mean(input).
        owners = {f"{n}.weight": m for n, m in module.named_modules()
                  if isinstance(m, torch.nn.Linear)}
        steerable = bool(self.steering) and all(key in owners for key, _ in resolved)
        act_sum: dict[str, Any] = {}
        handles = []
        if steerable:
            def make(key: str):
                def hook(_m: Any, args: Any) -> None:
                    x = args[0].detach()[0, -1].double().cpu()
                    act_sum[key] = x if key not in act_sum else act_sum[key] + x
                return hook
            handles = [owners[key].register_forward_pre_hook(make(key)) for key, _ in resolved]

        # Pass 1: unedited logits and the margin each case is scored on.
        kept: list[dict[str, Any]] = []
        skipped: Counter = Counter()
        try:
            for case in cases.stratified_head(self.max_cases):
                try:
                    clean = last_logits(case.inputs)
                except Exception:  # noqa: BLE001 - an unreadable case is counted, not fatal
                    skipped["forward_failed"] += 1
                    continue
                margin, why = build_margin(self.target, clean, model=model, case=case)
                if margin is None:
                    skipped[why] += 1
                    continue
                kept.append({"case": case, "clean": clean, "margin": margin,
                             "m0": margin.value(clean), "top1": int(clean[-1].argmax())})
        finally:
            for handle in handles:
                handle.remove()
        if not kept:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")
        n = len(kept)

        def measure(run: Any) -> "list[tuple[float, float, bool]]":
            """(margin change, KL from clean, top-1 flipped) per kept case under *run*."""
            out = []
            for item in kept:
                logits = run(item["case"].inputs)
                p0 = torch.log_softmax(item["clean"][-1].double(), dim=-1)
                p1 = torch.log_softmax(logits[-1].double(), dim=-1)
                kl = float((p0.exp() * (p0 - p1)).sum().clamp_min(0.0))
                out.append((item["margin"].value(logits) - item["m0"], kl,
                            item["margin"].top1_changed(logits, item["top1"])))
            return out

        with applied(model, self.edit) as report:
            applied_report = report.to_dict()
            edited = measure(last_logits)

        control_rows: dict[str, list[list[tuple[float, float, bool]]]] = {}
        control_errors: dict[str, str] = {}
        for kind in self.controls:
            for draw in range(int(self.n_seeds)):
                try:
                    control = build_control(model, self.edit, kind, seed=int(self.seed) + draw)
                except ValueError as exc:
                    control_errors[kind] = str(exc)
                    break
                with applied(model, control):
                    control_rows.setdefault(kind, []).append(measure(last_logits))

        steered = None
        if steerable:
            delta = self.edit.as_delta(module)
            shifts = {
                t.name: torch.from_numpy(t.delta).double() @ (act_sum[t.name] / n)
                for t in delta.tensors if t.name in act_sum
            }
            with output_shift(module, shifts):
                steered = measure(last_logits)

        per_case: list[dict[str, Any]] = []
        for i, item in enumerate(kept):
            d_edit, kl_edit, flip = edited[i]
            row: dict[str, Any] = {
                "sample_id": item["case"].id,
                "we_margin_clean": round(item["m0"], 4),
                "we_delta_edit": round(d_edit, 6),
                "we_kl_edit": round(kl_edit, 9),
                "we_flips_top1": bool(flip),
            }
            for kind, draws in control_rows.items():
                row[f"we_delta_{kind}"] = round(sum(d[i][0] for d in draws) / len(draws), 6)
                row[f"we_kl_{kind}"] = round(sum(d[i][1] for d in draws) / len(draws), 9)
            if steered is not None:
                row["we_delta_steer"] = round(steered[i][0], 6)
            per_case.append(row)

        # Summaries are computed from the unrounded measurements: the rows are
        # rounded for reading, and a ratio of two rounded small numbers is noise.
        def avg(values: "list[float]") -> float:
            return sum(values) / len(values)

        def control_mean(kind: str, field: int) -> float:
            draws = control_rows[kind]
            return avg([avg([d[i][field] for d in draws]) for i in range(n)])

        kl_edit_mean = avg([e[1] for e in edited])
        findings: dict[str, Any] = {
            "edit": self.edit.name,
            "n_cases_analyzed": n,
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "target": self.target,
            "max_relative_change": applied_report["max_relative_change"],
            "mean_delta_edit": avg([e[0] for e in edited]),
            "mean_kl_edit": kl_edit_mean,
            "frac_flips_top1": round(sum(bool(e[2]) for e in edited) / n, 4),
            "n_control_draws": int(self.n_seeds),
        }
        for kind in control_rows:
            kl_control = control_mean(kind, 1)
            findings[f"mean_kl_{kind}"] = kl_control
            findings[f"mean_delta_{kind}"] = control_mean(kind, 0)
            # Ratio of output movement: edit over matched control.
            findings[f"specificity_{kind}"] = (
                round(kl_edit_mean / kl_control, 4) if kl_control > 0 else None
            )
        if control_errors:
            findings["controls_not_run"] = control_errors
        if steered is not None:
            d_e = [e[0] for e in edited]
            d_s = [x[0] for x in steered]
            total = sum(x * x for x in d_e)
            findings["mean_delta_steer"] = avg(d_s)
            # 1 = the constant shift reproduces every per-case change; 0 = it
            # explains none of it; negative = it moves cases the other way.
            findings["steer_recovery"] = (
                round(1.0 - sum((a - b) ** 2 for a, b in zip(d_e, d_s)) / total, 4)
                if total > 0 else None
            )
        elif self.steering:
            findings["steering_not_run"] = (
                "an edited tensor is not the weight of a linear layer, so its input "
                "cannot be read to size the shift"
            )
        for label, tag in ((Label.FAIL, "fail"), (Label.PASS, "pass")):
            rows = [r for r, item in zip(per_case, kept) if item["case"].label == label]
            if rows:
                findings[f"n_{tag}"] = len(rows)
                findings[f"mean_delta_edit_{tag}"] = avg([r["we_delta_edit"] for r in rows])
                findings[f"frac_flips_{tag}"] = round(
                    sum(bool(r["we_flips_top1"]) for r in rows) / len(rows), 4)
        findings["structure"] = edit_structure(model, self.edit)["tensors"]
        findings["per_case"] = per_case
        findings["_caveat"] = (
            "Measured teacher-forced on the next-token distribution at the end of the "
            "prompt. Controls are random draws: with few draws or few cases a specificity "
            "ratio is an estimate, not a test. Effects that only appear during generation "
            "are not visible here."
        )
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
