"""Weight-edit continuation audit — what an edit does along a recorded generation.

:mod:`.edit_audit` measures an edit at one place, the end of the prompt. Some
edits act somewhere else: the edit studied in *Sparks of Weight-Space
Programming* does its work inside repetition loops that only appear thousands
of tokens into a generation. This analyzer follows the edit there.

Each case supplies a prompt (``case.inputs.prompt``) and the text the model
produced for it (``case.observed``). The two are concatenated and run once per
model state, teacher-forced, so every position of the continuation is scored on
the token that actually came next. Per position it records

* the change in negative log-likelihood of that next token (``dNLL`` > 0: the
  state makes the recorded continuation LESS likely), and
* the KL divergence from the unedited next-token distribution,

under the edit, under matched random-direction and random-neuron controls, and
under a constant activation shift with the weights untouched.

Positions are then split into two groups. With ``groups="repetition"`` a
prediction counts as in-loop when the ``repeat_ngram`` tokens before the
predicted token have already occurred earlier in the sequence, and as
productive otherwise; prompt positions are never scored. Reading the result:

* an edit that breaks loops raises the NLL of the repeated continuation at loop
  positions and leaves productive positions alone
  (``dnll_edit_loop_minus_productive`` well above zero);
* ``specificity_*`` near 1 in a group means a matched random change moves the
  output there as much as the edit does;
* ``steer_recovery_*`` near 1 means a constant activation shift reproduces the
  edit's per-position effect in that group.

Limits. This is teacher-forced on text the UNEDITED model wrote: it says what
the edit does to the probability of that text, not what the edited model would
write instead. The prompt is tokenised by the model handle and the continuation
is appended as plain tokens, so the handle must be set up to produce exactly
the prompt the generation was made from (chat template included). Text-only
cases only. Memory: one pass holds two ``[tokens, vocab]`` logit tensors.

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

GROUPINGS = ("repetition", "none")


def repetition_mask(ids: "list[int]", n: int) -> "list[bool]":
    """``mask[i]`` is True when the *n* tokens before index *i* occurred earlier.

    The window ``ids[i-n:i]`` is compared with every earlier window of the same
    length, so ``mask[i]`` describes the context in which token *i* is
    predicted: inside a stretch the sequence has already produced once.
    """
    seen: set[tuple[int, ...]] = set()
    mask = [False] * len(ids)
    for i in range(n, len(ids)):
        window = tuple(ids[i - n:i])
        mask[i] = window in seen
        seen.add(window)
    return mask


def _continuation_text(case: Any) -> "str | None":
    observed = getattr(case, "observed", None)
    if observed is None:
        return None
    text = observed if isinstance(observed, str) else str(observed)
    return text if text.strip() else None


@register_analyzer("weight_edit_continuation_audit")
class WeightEditContinuationAudit(Analyzer):
    """Follow a weight edit along recorded generations: loop vs productive positions, against controls.

    Args:
        edit:         The :class:`~evalrx.wsp.edit.WeightEdit` to audit.
        groups:       ``"repetition"`` splits scored positions into ``loop`` and
                      ``productive`` (see module docstring); ``"none"`` keeps
                      them in one group, ``all``.
        repeat_ngram: Window length of the repetition test.
        controls:     Control kinds to run (see :mod:`evalrx.wsp.controls`).
        n_seeds:      Random draws per control kind.
        steering:     Whether to run the constant-shift steering control.
        max_tokens:   Truncate prompt + continuation to this many tokens.
        chunk:        Positions scored at a time (bounds the float32 memory).
        seed:         Base seed of the control draws.
        max_cases:    Analyse at most this many cases (0 = all).
    """

    name = "weight_edit_continuation_audit"
    requires = frozenset({Capability.LOGITS})
    question = ("Where along a generation does the edit act, and does a matched random "
                "change act there too?")
    signal_docs = {
        "wc_frac_loop": (
            "Share of loop positions",
            "Share of this generation's scored positions that sit inside a repeated "
            "stretch. High means the model was looping.",
        ),
        "wc_dnll_edit_loop": (
            "Loop text made less likely",
            "How much less likely the edit makes the recorded next token at positions "
            "inside a repeated stretch (change in negative log-likelihood). Positive "
            "means the edit pushes the model off the loop.",
        ),
        "wc_dnll_edit_productive": (
            "Normal text made less likely",
            "The same change at positions outside any repeated stretch. Near zero means "
            "the edit leaves ordinary text alone; positive is a side effect.",
        ),
        "wc_kl_edit_loop": (
            "Output shift in loops",
            "How far the edit moves the next-token distribution at loop positions "
            "(KL divergence from the unedited model).",
        ),
        "wc_kl_edit_productive": (
            "Output shift elsewhere",
            "How far the edit moves the next-token distribution at positions outside "
            "loops. Large values mean the edit also changes ordinary text.",
        ),
    }

    def __init__(self, edit: "Any", groups: str = "repetition", repeat_ngram: int = 32,
                 controls: "tuple[str, ...]" = ("random_direction", "random_columns"),
                 n_seeds: int = 1, steering: bool = True, max_tokens: int = 4096,
                 chunk: int = 256, seed: int = 0, max_cases: int = 0) -> None:
        super().__init__(edit=edit, groups=groups, repeat_ngram=repeat_ngram,
                         controls=tuple(controls), n_seeds=n_seeds, steering=steering,
                         max_tokens=max_tokens, chunk=chunk, seed=seed, max_cases=max_cases)

    # ------------------------------------------------------------------
    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        import torch

        from evalrx.analyzers._internals.access import media_positions, tokenizer_of
        from evalrx.wsp.controls import CONTROL_KINDS, build_control
        from evalrx.wsp.direction import output_shift
        from evalrx.wsp.edit import applied, module_of, resolve_parameters
        from evalrx.wsp.structure import edit_structure

        if self.groups not in GROUPINGS:
            raise ValueError(f"{self.name}: groups must be one of {GROUPINGS}, got {self.groups!r}")
        unknown = [k for k in self.controls if k not in CONTROL_KINDS]
        if unknown:
            raise ValueError(f"{self.name}: unknown control(s) {unknown}; choose from {CONTROL_KINDS}")
        if int(self.n_seeds) < 1 or int(self.repeat_ngram) < 1 or int(self.chunk) < 1:
            raise ValueError(f"{self.name}: n_seeds, repeat_ngram and chunk must be positive")
        if int(self.max_tokens) < 3:
            raise ValueError(f"{self.name}: max_tokens must be at least 3")
        module = module_of(model)
        tokenizer = tokenizer_of(model)
        if tokenizer is None:
            raise ValueError(f"{self.name}: the model handle exposes no tokenizer to encode "
                             "the continuation with")
        resolved = resolve_parameters(module, self.edit)
        device = next(module.parameters()).device
        group_names = ("loop", "productive") if self.groups == "repetition" else ("all",)

        def logits_of(ids: "list[int]") -> Any:
            with torch.no_grad():
                out = module(input_ids=torch.tensor([ids], device=device))
            return getattr(out, "logits", out)[0]

        # ---- Pass 1: build each case's token sequence and group labels -------------
        items: list[dict[str, Any]] = []
        skipped: Counter = Counter()
        for case in cases.stratified_head(self.max_cases):
            text = _continuation_text(case)
            if text is None:
                skipped["no_continuation"] += 1
                continue
            try:
                trace = model.forward(case.inputs, capture={Capability.LOGITS})
            except Exception:  # noqa: BLE001 - an unreadable case is counted, not fatal
                skipped["forward_failed"] += 1
                continue
            if media_positions(trace)[1]:
                skipped["has_media"] += 1
                continue
            prompt_ids = [int(t) for t in trace.token_ids]
            try:
                cont = tokenizer(text, add_special_tokens=False)["input_ids"]
            except Exception:  # noqa: BLE001
                skipped["continuation_untokenizable"] += 1
                continue
            if cont and isinstance(cont[0], (list, tuple)):
                cont = cont[0]
            ids = (prompt_ids + [int(t) for t in cont])[: int(self.max_tokens)]
            start = len(prompt_ids)          # index of the first continuation token
            if len(ids) <= start:
                skipped["no_continuation_tokens_after_truncation"] += 1
                continue
            # Target index i (predicted from position i - 1) for every continuation token.
            targets = list(range(max(start, 1), len(ids)))
            if self.groups == "repetition":
                mask = repetition_mask(ids, int(self.repeat_ngram))
                labels = ["loop" if mask[i] else "productive" for i in targets]
            else:
                labels = ["all"] * len(targets)
            items.append({"case": case, "ids": ids, "targets": targets, "labels": labels})
        if not items:
            raise ValueError(f"{self.name}: no case could be analysed (skipped: {dict(skipped)})")

        # ---- Controls, built once against the unedited weights ---------------------
        states: list[tuple[str, Any]] = [("edit", self.edit)]
        control_errors: dict[str, str] = {}
        for kind in self.controls:
            for draw in range(int(self.n_seeds)):
                try:
                    states.append((kind, build_control(model, self.edit, kind,
                                                       seed=int(self.seed) + draw)))
                except ValueError as exc:
                    control_errors[kind] = str(exc)
                    break

        # ---- Steering shift: the edit's mean effect over all scored positions -------
        owners = {f"{n}.weight": m for n, m in module.named_modules()
                  if isinstance(m, torch.nn.Linear)}
        steerable = bool(self.steering) and all(key in owners for key, _ in resolved)
        shifts: "dict[str, Any] | None" = None
        if steerable:
            act_sum: dict[str, Any] = {}
            current: dict[str, Any] = {}

            def make(key: str):
                def hook(_m: Any, args: Any) -> None:
                    x = args[0].detach()[0].double()          # [tokens, d_in]
                    total = x[current["positions"]].sum(dim=0).cpu()
                    act_sum[key] = total if key not in act_sum else act_sum[key] + total
                return hook

            handles = [owners[key].register_forward_pre_hook(make(key)) for key, _ in resolved]
            n_scored = 0
            try:
                for item in items:
                    # The input that produces the prediction of target i sits at i - 1.
                    current["positions"] = torch.tensor([i - 1 for i in item["targets"]],
                                                        device=device)
                    logits_of(item["ids"])
                    n_scored += len(item["targets"])
            finally:
                for handle in handles:
                    handle.remove()
            delta = self.edit.as_delta(module)
            shifts = {t.name: torch.from_numpy(t.delta).double() @ (act_sum[t.name] / n_scored)
                      for t in delta.tensors}

        # ---- Pass 2: per case, the clean run and every state ------------------------
        def score(clean: Any, other: Any, item: dict[str, Any]) -> "tuple[Any, Any, Any, Any]":
            """Per target: clean NLL, NLL change, KL from clean, top-1 changed."""
            pos = torch.tensor([i - 1 for i in item["targets"]], device=device)
            tgt = torch.tensor([item["ids"][i] for i in item["targets"]], device=device)
            out = [[], [], [], []]
            for a in range(0, len(pos), int(self.chunk)):
                p, t = pos[a:a + int(self.chunk)], tgt[a:a + int(self.chunk)]
                lp0 = torch.log_softmax(clean[p].float(), dim=-1)
                lp1 = torch.log_softmax(other[p].float(), dim=-1)
                nll0 = -lp0.gather(1, t[:, None])[:, 0]
                nll1 = -lp1.gather(1, t[:, None])[:, 0]
                out[0].append(nll0.double().cpu())
                out[1].append((nll1 - nll0).double().cpu())
                out[2].append((lp0.exp() * (lp0 - lp1)).sum(dim=-1).clamp_min(0).double().cpu())
                out[3].append((lp0.argmax(dim=-1) != lp1.argmax(dim=-1)).cpu())
            return tuple(torch.cat(part) for part in out)

        state_names = ["edit"] + [k for k in self.controls if k not in control_errors]
        if shifts is not None:
            state_names.append("steer")
        # pooled[state][group] -> lists of per-position tensors across cases
        pooled: dict[str, dict[str, dict[str, list]]] = {
            s: {g: {"dnll": [], "kl": [], "flip": []} for g in group_names} for s in state_names}
        clean_nll: dict[str, list] = {g: [] for g in group_names}
        per_case: list[dict[str, Any]] = []
        applied_report: "dict[str, Any] | None" = None

        for item in items:
            clean = logits_of(item["ids"])
            index = {g: torch.tensor([j for j, lab in enumerate(item["labels"]) if lab == g],
                                     dtype=torch.long) for g in group_names}
            row: dict[str, Any] = {"sample_id": item["case"].id,
                                   "wc_n_positions": len(item["targets"])}
            if self.groups == "repetition":
                row["wc_frac_loop"] = round(len(index["loop"]) / len(item["targets"]), 4)
            by_state: dict[str, list] = {}
            for label, variant in states:
                with applied(model, variant) as report:
                    if label == "edit" and applied_report is None:
                        applied_report = report.to_dict()
                    by_state.setdefault(label, []).append(score(clean, logits_of(item["ids"]), item))
            if shifts is not None:
                with output_shift(module, shifts):
                    by_state["steer"] = [score(clean, logits_of(item["ids"]), item)]
            del clean

            first = by_state["edit"][0]
            for g in group_names:
                if len(index[g]):
                    clean_nll[g].append(first[0][index[g]])
                    row[f"wc_nll_clean_{g}"] = round(float(first[0][index[g]].mean()), 6)
            for label, draws in by_state.items():
                # Average the draws position by position, then split by group.
                dnll = torch.stack([d[1] for d in draws]).mean(dim=0)
                kl = torch.stack([d[2] for d in draws]).mean(dim=0)
                flip = torch.stack([d[3].double() for d in draws]).mean(dim=0)
                for g in group_names:
                    if not len(index[g]):
                        continue
                    pooled[label][g]["dnll"].append(dnll[index[g]])
                    pooled[label][g]["kl"].append(kl[index[g]])
                    pooled[label][g]["flip"].append(flip[index[g]])
                    row[f"wc_dnll_{label}_{g}"] = round(float(dnll[index[g]].mean()), 6)
                    row[f"wc_kl_{label}_{g}"] = round(float(kl[index[g]].mean()), 9)
            per_case.append(row)

        # ---- Summaries over positions ------------------------------------------------
        def cat(parts: list) -> Any:
            return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

        findings: dict[str, Any] = {
            "edit": self.edit.name,
            "n_cases_analyzed": len(items),
            "n_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
            "groups": self.groups,
            "repeat_ngram": int(self.repeat_ngram) if self.groups == "repetition" else None,
            "max_tokens": int(self.max_tokens),
            "n_control_draws": int(self.n_seeds),
            "max_relative_change": (applied_report or {}).get("max_relative_change"),
            "n_positions": sum(len(i["targets"]) for i in items),
        }
        for g in group_names:
            nll0 = cat(clean_nll[g])
            findings[f"n_positions_{g}"] = int(nll0.numel())
            if not nll0.numel():
                continue
            findings[f"mean_nll_clean_{g}"] = float(nll0.mean())
            edit_dnll = cat(pooled["edit"][g]["dnll"])
            edit_kl = float(cat(pooled["edit"][g]["kl"]).mean())
            for label in state_names:
                dnll = cat(pooled[label][g]["dnll"])
                kl = float(cat(pooled[label][g]["kl"]).mean())
                findings[f"mean_dnll_{label}_{g}"] = float(dnll.mean())
                findings[f"mean_kl_{label}_{g}"] = kl
                findings[f"frac_flips_{label}_{g}"] = round(
                    float(cat(pooled[label][g]["flip"]).mean()), 4)
                if label in self.controls:
                    # Output movement of the edit over that of the matched control.
                    findings[f"specificity_{label}_{g}"] = round(edit_kl / kl, 4) if kl > 0 else None
                if label == "steer":
                    total = float((edit_dnll ** 2).sum())
                    findings[f"steer_recovery_{g}"] = (
                        round(1.0 - float(((edit_dnll - dnll) ** 2).sum()) / total, 4)
                        if total > 0 else None)
        if self.groups == "repetition" and all(
                f"mean_dnll_edit_{g}" in findings for g in group_names):
            findings["dnll_edit_loop_minus_productive"] = (
                findings["mean_dnll_edit_loop"] - findings["mean_dnll_edit_productive"])
        if control_errors:
            findings["controls_not_run"] = control_errors
        if self.steering and shifts is None:
            findings["steering_not_run"] = (
                "an edited tensor is not the weight of a linear layer, so its input "
                "cannot be read to size the shift")
        findings["structure"] = edit_structure(model, self.edit)["tensors"]
        findings["per_case"] = per_case
        findings["_caveat"] = (
            "Teacher-forced on text the unedited model wrote: this measures what each "
            "state does to the probability of that text, not what an edited model would "
            "write. Positions within one generation are not independent, so the "
            "per-position means carry less evidence than their count suggests; compare "
            "across cases (per_case) before drawing a conclusion. Controls are random "
            "draws."
        )
        return Result(analyzer=self.name, model=repr(model), cases=cases, findings=findings)
