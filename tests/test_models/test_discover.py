"""Runtime module-path discovery — the anti-hardcoding mechanism."""

from __future__ import annotations

import pytest
import torch.nn as nn

from evalrx.models._discover import find_decoder_layers, get_unembed, resolve


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(4, 4)


class _Inner(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])


class _Cfg:
    def __init__(self, n: int) -> None:
        self.num_hidden_layers = n


class _Top(nn.Module):
    """Mimics the post-refactor single-.model wrapper."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.model = _Inner(n)
        self.lm_head = nn.Linear(4, 10)
        self.config = _Cfg(n)

    def get_output_embeddings(self):
        return self.lm_head


def test_finds_decoder_layers_by_length_and_shape():
    top = _Top(3)
    layers, path = find_decoder_layers(top, num_hidden_layers=3)
    assert path == "model.layers"
    assert len(layers) == 3


def test_reads_layer_count_from_config_when_not_given():
    top = _Top(5)
    layers, path = find_decoder_layers(top)  # infers n=5 from config
    assert len(layers) == 5 and path == "model.layers"


def test_get_unembed_returns_lm_head():
    top = _Top()
    assert get_unembed(top) is top.lm_head


def test_resolve_path_errors_clearly():
    top = _Top()
    assert resolve(top, "model.layers") is top.model.layers
    with pytest.raises(AttributeError):
        resolve(top, "model.nonexistent")


def test_raises_when_no_decoder_list_found():
    empty = nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        find_decoder_layers(empty, num_hidden_layers=3)
