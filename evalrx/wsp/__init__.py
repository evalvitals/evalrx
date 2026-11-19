"""Weight-space programming: build, apply, control and inspect direct weight edits.

    from evalrx.wsp import WeightEdit, TensorEdit, applied, edit_structure

    edit = WeightEdit([TensorEdit.rank_one("layers.15.mlp.down_proj.weight", u, alpha=6.0,
                                           cols=columns)])
    edit_structure(model, edit)             # what changed, read from the weights
    with applied(model, edit):              # written in, then restored exactly
        answer = model.generate(case.inputs)

The behavioural audit lives with the analyzers
(:mod:`evalrx.analyzers.weights`), and the repair executor with the fix stage
(:mod:`evalrx.eval_agent.stages.fix.weight_edit`).
"""

from evalrx.wsp.continuation import (
    ContinuationSequence,
    ContinuationStats,
    encode_continuation,
    encode_continuations,
    repetition_mask,
)
from evalrx.wsp.controls import CONTROL_KINDS, build_control
from evalrx.wsp.direction import (
    WriteSite,
    collect_continuation_stats,
    collect_write_stats,
    continuation_contrast_edit,
    contrast_edit,
    edit_from_stats,
    output_shift,
    rank_columns,
    write_sites,
)
from evalrx.wsp.edit import (
    AppliedEdit,
    TensorEdit,
    WeightEdit,
    applied,
    bf16_bits_to_f32,
    module_of,
    resolve_parameters,
    tensor_sha256,
)
from evalrx.wsp.structure import delta_structure, edit_agreement, edit_structure, fit_rank_one

__all__ = [
    "CONTROL_KINDS",
    "AppliedEdit",
    "ContinuationSequence",
    "ContinuationStats",
    "TensorEdit",
    "WeightEdit",
    "WriteSite",
    "applied",
    "bf16_bits_to_f32",
    "build_control",
    "collect_continuation_stats",
    "collect_write_stats",
    "continuation_contrast_edit",
    "contrast_edit",
    "delta_structure",
    "edit_agreement",
    "edit_from_stats",
    "edit_structure",
    "encode_continuation",
    "encode_continuations",
    "fit_rank_one",
    "module_of",
    "output_shift",
    "rank_columns",
    "repetition_mask",
    "resolve_parameters",
    "tensor_sha256",
    "write_sites",
]
