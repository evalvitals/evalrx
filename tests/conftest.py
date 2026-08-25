"""Shared test fixtures: a fully-mocked Qwen that needs no weights or GPU."""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.model import Model, Trace

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the box
    # Only ``FakeModel``'s tensor methods need torch. Importing it at module
    # scope made a multi-gigabyte dependency a precondition for collecting ANY
    # test, so a machine without it could not run the contract or reporting
    # suites -- which are pure Python and have no tensors in them at all.
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: "ModuleNotFoundError | None" = exc
else:
    _TORCH_IMPORT_ERROR = None


from evalvitals.core.capability import Capability
from evalvitals.core.model import Model, Trace


def _torch():
    """torch, or a clear error naming what actually needs it."""
    if torch is None:
        raise RuntimeError(
            "this test drives FakeModel's tensor path and needs torch installed"
        ) from _TORCH_IMPORT_ERROR
    return torch


class FakeModel(Model):
    """A minimal in-memory Model for tests.

    Declares a configurable capability set and returns a deterministic Trace
    from ``forward`` — no HuggingFace, no GPU.
    """

    def __init__(
        self,
        capabilities: set[Capability] | None = None,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 5,
        hidden_dim: int = 8,
        vocab: int = 32,
        modalities: set[str] | None = None,
    ) -> None:
        self.capabilities = frozenset(
            capabilities
            if capabilities is not None
            else {Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES}
        )
        self.modalities = frozenset(modalities or {"text"})
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._seq_len = seq_len
        self._hidden_dim = hidden_dim
        self._vocab = vocab

    def generate(self, inputs, **kwargs) -> str:
        return "fake-output"

    def unembed_weight(self):
        _torch().manual_seed(1)
        return _torch().rand(self._vocab, self._hidden_dim)

    def logprobs(self, inputs, **kwargs):
        from evalvitals.core.model import TokenLogprob

        return [
            TokenLogprob(token=f"w{i}", logprob=-0.1 * (i + 1),
                         top={f"w{i}": -0.1 * (i + 1), "alt": -2.5})
            for i in range(4)
        ]

    def forward(self, inputs, capture: set[Capability], spec=None) -> Trace:
        _torch().manual_seed(0)
        provided: set[Capability] = set()
        attentions = hidden_states = logits = None
        if Capability.ATTENTION in capture and Capability.ATTENTION in self.capabilities:
            attentions = [
                _torch().rand(self._n_heads, self._seq_len, self._seq_len)
                for _ in range(self._n_layers)
            ]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture and Capability.HIDDEN_STATES in self.capabilities:
            hidden_states = [
                _torch().rand(self._seq_len, self._hidden_dim) for _ in range(self._n_layers + 1)
            ]
            provided.add(Capability.HIDDEN_STATES)
        if Capability.LOGITS in capture and Capability.LOGITS in self.capabilities:
            logits = _torch().rand(self._seq_len, self._vocab)
            provided.add(Capability.LOGITS)
        return Trace(
            tokens=[f"t{i}" for i in range(self._seq_len)],
            token_ids=list(range(self._seq_len)),
            provided=provided,
            attentions=attentions,
            hidden_states=hidden_states,
            logits=logits,
        )

    def __repr__(self) -> str:
        return f"FakeModel(caps={sorted(c.value for c in self.capabilities)})"


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", default=False, help="run GPU integration tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        # If --run-gpu is passed, verify CUDA is available
        if not _torch().cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="--run-gpu specified but no CUDA GPU is available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
        return

    # Skip all GPU/heavy tests by default
    skip_gpu = pytest.mark.skip(reason="GPU tests skipped by default. Pass --run-gpu to run them.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
