"""The ``api`` backend: hosted models behind a generate / chat function.

``backend`` holds :class:`APIBackend` / :class:`APIModel`; ``openai`` and
``gemini`` build those functions for OpenAI-compatible endpoints and the Google
GenAI SDK (``gemini_model`` is the underlying Gemini client). Everything
imports without the provider SDKs installed; they load lazily.
"""

from evalrx.models.backends.api.backend import (
    APIBackend,
    APIModel,
    call_vision_api_chat_fn,
    call_vision_api_generate_fn,
    parse_openai_logprobs,
)

__all__ = [
    "APIBackend",
    "APIModel",
    "call_vision_api_chat_fn",
    "call_vision_api_generate_fn",
    "parse_openai_logprobs",
]
