"""White-box (local) model entry points — per-version convenience factories.

Identity lives in ``evalrx.specs``; these are thin wrappers over
``compose(spec, "hf_local")``.  Import a specific version factory, e.g.::

    from evalrx.models.whitebox import qwen3_8b, qwen3_vl_8b_instruct, qwen3_omni_30b_a3b_instruct
"""

from evalrx.models.whitebox import qwen as _qwen
from evalrx.models.whitebox import qwen2_5_omni as _qwen2_5_omni
from evalrx.models.whitebox import qwen2_audio as _qwen2_audio
from evalrx.models.whitebox import qwen_omni as _qwen_omni
from evalrx.models.whitebox import qwen_vl as _qwen_vl
from evalrx.models.whitebox.qwen import *  # noqa: F401,F403  (QwenLLM + qwen text factories)
from evalrx.models.whitebox.qwen2_5_omni import *  # noqa: F401,F403  (qwen2.5-omni factory)
from evalrx.models.whitebox.qwen2_audio import *  # noqa: F401,F403  (qwen2-audio factory)
from evalrx.models.whitebox.qwen_omni import *  # noqa: F401,F403  (qwen3-omni factories)
from evalrx.models.whitebox.qwen_vl import *  # noqa: F401,F403  (QwenVL + qwen-vl factories)

__all__ = sorted(
    set(_qwen.__all__)
    | set(_qwen_vl.__all__)
    | set(_qwen_omni.__all__)
    | set(_qwen2_5_omni.__all__)
    | set(_qwen2_audio.__all__)
)
