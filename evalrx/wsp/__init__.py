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

from evalrx.wsp.controls import CONTROL_KINDS, build_control
from evalrx.wsp.direction import (
    WriteSite,
    collect_write_stats,
    contrast_edit,
    output_shift,
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
from evalrx.wsp.structure import delta_structure, edit_structure

__all__ = [
    "CONTROL_KINDS",
    "AppliedEdit",
    "TensorEdit",
    "WeightEdit",
    "WriteSite",
    "applied",
    "bf16_bits_to_f32",
    "build_control",
    "collect_write_stats",
    "contrast_edit",
    "delta_structure",
    "edit_structure",
    "module_of",
    "output_shift",
    "resolve_parameters",
    "tensor_sha256",
    "write_sites",
]
