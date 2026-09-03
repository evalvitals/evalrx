"""TruthX-backed IFCD components.

IFCD's paper contrasts an anti-hallucination and a hallucination-inducing
internal-representation edit.  This module ports the released TruthX MLP
editor and exposes it through reversible modern-HF hooks.  The hooks edit the
post-projection attention output (the current Transformers boundary), whereas
the original LLaMA patch edits just before ``o_proj``; callers must therefore
report this route as an architecture adaptation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _dims(value: Any) -> list[int]:
    text = str(value or "")
    return [int(item) for item in text.split(",") if item]


def _checkpoint_arg(args: Any, name: str) -> Any:
    return args.get(name) if isinstance(args, dict) else getattr(args, name)


class TruthXEditor:
    """Released TruthX autoencoder with a reversible edit-strength switch."""

    def __init__(self, checkpoint_path: str | Path, *, hidden_size: int, top_layers: int = 15):
        import torch
        from torch import nn

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"TruthX checkpoint does not exist: {path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        args = checkpoint["args"]
        semantic_dim = int(_checkpoint_arg(args, "semantic_latent_dim"))
        truthful_dim = int(_checkpoint_arg(args, "truthful_latent_dim"))

        def stack(in_dim: int, hidden: list[int], out_dim: int, *, normalize: bool) -> Any:
            modules: list[Any] = []
            current = in_dim
            for dim in hidden:
                modules.append(nn.Sequential(nn.Linear(current, dim), nn.LayerNorm(dim), nn.LeakyReLU()))
                current = dim
            modules.append(nn.Sequential(nn.Linear(current, out_dim), nn.LayerNorm(out_dim), nn.LeakyReLU()))
            return nn.Sequential(*modules)

        class Autoencoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.semantic_encoder = stack(
                    hidden_size, _dims(_checkpoint_arg(args, "semantic_hidden_dims")), semantic_dim, normalize=False
                )
                self.truthful_encoder = stack(
                    hidden_size, _dims(_checkpoint_arg(args, "truthful_hidden_dims")), truthful_dim, normalize=True
                )
                self.proj = nn.Linear(truthful_dim, semantic_dim, bias=False) if semantic_dim != truthful_dim else None
                decoder_dims = _dims(_checkpoint_arg(args, "decoder_hidden_dims"))
                decoder: list[Any] = []
                current = semantic_dim
                for dim in decoder_dims:
                    decoder.append(nn.Sequential(nn.Linear(current, dim), nn.LayerNorm(dim), nn.LeakyReLU()))
                    current = dim
                self.decoder = nn.Sequential(*decoder) if decoder else nn.Identity()
                # TruthX stores this as ``nn.Sequential(Linear(...))``;
                # preserve the state-dict key ``final_layer.0`` exactly.
                self.final_layer = nn.Sequential(nn.Linear(current, hidden_size))
                self.cross_attention = nn.MultiheadAttention(embed_dim=semantic_dim, num_heads=1)

            def truth(self, x: Any) -> Any:
                return torch.nn.functional.normalize(self.truthful_encoder(x), p=2, dim=-1)

            def reconstruct(self, x: Any, truth: Any) -> Any:
                semantic = self.semantic_encoder(x)
                if self.proj is not None:
                    truth = self.proj(truth)
                attended, _ = self.cross_attention(
                    semantic.unsqueeze(0), truth.unsqueeze(0), truth.unsqueeze(0)
                )
                return self.final_layer(self.decoder(semantic + attended[0]))

        self.model = Autoencoder().to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.pos_center = checkpoint["pos_center"].to(device)
        self.neg_center = checkpoint["neg_center"].to(device)
        # The official release serializes this as a Python list, unlike the
        # center tensors. Keep it host-side because it only gates hook setup.
        self.rank = [int(value) for value in checkpoint["rank"]]
        self.top_layers = int(top_layers)
        self.strength = 0.5

    def edit(self, values: Any, *, layer_index: int, component: str) -> Any:
        """Apply TruthX's positive or negative edit only to the final token."""
        import torch
        import torch.nn.functional as F

        rank_index = 2 * int(layer_index) + (0 if component == "attn" else 1)
        if rank_index >= len(self.rank) or int(self.rank[rank_index]) > self.top_layers:
            return values
        batch, sequence, width = values.shape
        flat = values.contiguous().view(-1, width)
        dtype = self.model.semantic_encoder[0][0].weight.dtype
        x = flat.to(dtype)
        # TruthX encodes the unflattened sequence, then flattens it inside its
        # autoencoder forward. Keeping that order is material: its
        # cross-attention is over tokens in the current generated sequence.
        truth = self.model.truth(values.to(dtype))
        pos = self.pos_center[rank_index].unsqueeze(0).to(dtype)
        neg = self.neg_center[rank_index].unsqueeze(0).to(dtype)
        direction = (pos - neg).unsqueeze(0)
        positive = self.model.reconstruct(
            x, F.normalize(truth + direction, p=2, dim=-1).reshape(-1, truth.shape[-1])
        )
        negative = self.model.reconstruct(
            x, F.normalize(truth - direction, p=2, dim=-1).reshape(-1, truth.shape[-1])
        )
        delta = (positive - negative).view(batch, sequence, width).to(values.dtype)
        delta = F.normalize(delta, p=2, dim=-1) * torch.linalg.vector_norm(
            values, ord=2, dim=-1, keepdim=True
        )
        mask = torch.zeros((batch, sequence, 1), device=values.device, dtype=values.dtype)
        mask[:, -1:, :] = 1
        return values + delta.to(values.dtype) * float(self.strength) * mask


@contextmanager
def truthx_editing(language_model: Any, editor: TruthXEditor) -> Iterator[None]:
    """Install temporary TruthX hooks on the LLaMA attention and MLP outputs."""
    decoder = getattr(language_model, "model", language_model)
    handles: list[Any] = []

    def edit_output(layer_index: int, component: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                return (editor.edit(output[0], layer_index=layer_index, component=component), *output[1:])
            return editor.edit(output, layer_index=layer_index, component=component)

        return hook

    try:
        for index, layer in enumerate(decoder.layers):
            handles.append(layer.self_attn.register_forward_hook(edit_output(index, "attn")))
            handles.append(layer.mlp.register_forward_hook(edit_output(index, "ffn")))
        yield
    finally:
        for handle in handles:
            handle.remove()


class IFCDLogitsProcessor:
    """Compute ``(1 + alpha) log p+ - alpha log p-`` with IFCD's cutoff."""

    def __init__(
        self,
        model: Any,
        negative_inputs: dict[str, Any],
        editor: TruthXEditor,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        edit_strength: float = 0.5,
    ) -> None:
        self.model = model
        self.negative_inputs = negative_inputs
        self.editor = editor
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.edit_strength = float(edit_strength)
        self._negative_output: Any = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import math

        import torch
        import torch.nn.functional as F

        positive = F.log_softmax(scores, dim=-1)
        old_strength = self.editor.strength
        self.editor.strength = -self.edit_strength
        try:
            with torch.no_grad():
                if self._negative_output is None:
                    self._negative_output = self.model(
                        **self.negative_inputs, use_cache=True, return_dict=True
                    )
                else:
                    self._negative_output = self.model(
                        input_ids=input_ids[:, -1:],
                        past_key_values=self._negative_output.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
        finally:
            self.editor.strength = old_strength
        negative = F.log_softmax(self._negative_output.logits[:, -1, :], dim=-1)
        cutoff = positive.max(dim=-1, keepdim=True).values + math.log(self.beta)
        contrastive = (1 + self.alpha) * positive - self.alpha * negative
        return contrastive.masked_fill(positive < cutoff, -float("inf"))
