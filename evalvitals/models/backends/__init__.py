"""Backends — runtimes that turn a ModelSpec into a Model.

Capabilities come from the backend; identity from the spec.  All three import
torch-free (heavy deps are lazy inside ``build``/``load``), so the registry and
``compose()`` work on the light install.
"""

from evalvitals.models.backends.api import (
    APIBackend,
    APIModel,
    call_vision_api_chat_fn,
    call_vision_api_generate_fn,
    parse_openai_logprobs,
)
from evalvitals.models.backends.base import Backend, RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalBackend, HFLocalModel
from evalvitals.models.backends.openai_compat import (
    openai_chat_fn,
    openai_generate_fn,
    openai_runtime,
)
from evalvitals.models.backends.vllm_offline import VLLMOfflineBackend

#: name -> backend class.  Plain dict (no import-side-effect decorator).
BACKENDS: dict[str, type[Backend]] = {
    "api": APIBackend,
    "hf_local": HFLocalBackend,
    "vllm_offline": VLLMOfflineBackend,
}

__all__ = [
    "Backend",
    "RuntimeConfig",
    "APIBackend",
    "APIModel",
    "HFLocalBackend",
    "HFLocalModel",
    "VLLMOfflineBackend",
    "call_vision_api_generate_fn",
    "call_vision_api_chat_fn",
    "parse_openai_logprobs",
    "openai_chat_fn",
    "openai_generate_fn",
    "openai_runtime",
    "BACKENDS",
]
