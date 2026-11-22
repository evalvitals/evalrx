"""What the run logger puts on disk beside its JSON: atomic document writes,
sandbox workspaces inlined into an event, and numeric artifacts saved as
``.npy`` with a rendered figure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Extensions treated as genuine binary media — the one thing the run log
#: keeps as a separate file. Everything else becomes a JSON string value.
_MEDIA_EXTS = frozenset({
    ".npy", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".wav", ".mp3", ".flac", ".ogg", ".mp4", ".avi", ".mov",
})

#: Text/code file suffixes worth inlining from a sandbox workspace snapshot
#: (skip weights/binaries — those go through the media/artifact path instead).
_INLINE_SUFFIXES = frozenset(
    {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".log", ".toml"}
)
_INLINE_MAX_BYTES = 2_000_000  # skip (note-only) any single file larger than this


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write *obj* as JSON to *path* such that a reader never sees a partial file.

    Writes to a sibling temp file first, then ``os.replace`` (atomic on the
    same filesystem) — a crash mid-write leaves the OLD complete file in
    place, never a truncated one. This runs on every single logged event
    (see the design doc's "durability" section for the cost trade-off that
    was chosen deliberately here, not overlooked).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def inline_workspace(
    workdir: "str | Path", media_dir: Path, *, run_dir: "Path | None" = None,
    max_bytes: "int | None" = _INLINE_MAX_BYTES,
    preserve_all: bool = False,
) -> "dict[str, Any] | None":
    """Read a sandbox working directory into a JSON-safe dict, inlining text.

    Returns ``{"files": {relative_path: content_or_note}, "media": [rel_paths],
    "skipped": n}`` or ``None`` when *workdir* does not exist. Text-like files
    (see ``_INLINE_SUFFIXES``) are read and inlined verbatim; recognised media
    extensions are COPIED into *media_dir* (a real binary artifact, rule 4's
    one exception) with a path reference left in ``"media"``; anything else is
    skipped with a one-line note so its existence is still visible.
    """
    import hashlib
    import shutil

    src = Path(workdir)
    if not src.exists() or not src.is_dir():
        return None
    files: dict[str, Any] = {}
    media: list[str] = []
    media_files: dict[str, str] = {}
    skipped = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = str(f.relative_to(src))
        suffix = f.suffix.lower()
        inline_content = None
        binary = suffix in _MEDIA_EXTS
        if preserve_all and suffix not in _INLINE_SUFFIXES and not binary:
            try:
                inline_content = f.read_text(encoding="utf-8")
                if "\x00" in inline_content:
                    binary = True
            except UnicodeError:
                binary = True
        if binary:
            media_dir.mkdir(parents=True, exist_ok=True)
            # Path + content identify a snapshot: trials cannot collide, and
            # later writes to the same source cannot replace earlier evidence.
            hasher = hashlib.sha256(str(f.resolve()).encode("utf-8"))
            with f.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()[:12]
            dest = media_dir / f"{f.stem}_{digest}{suffix}"
            try:
                shutil.copy2(f, dest)
                try:
                    media.append(str(dest.relative_to(run_dir)) if run_dir else str(dest))
                except ValueError:
                    media.append(str(dest))
                media_files[rel] = media[-1]
            except Exception:  # noqa: BLE001
                if preserve_all:
                    raise  # do not let finalization delete unarchived evidence
                skipped += 1
            continue
        if suffix not in _INLINE_SUFFIXES and inline_content is None:
            files[rel] = f"<skipped: {suffix or 'no extension'}, not a recognised text type>"
            skipped += 1
            continue
        try:
            if max_bytes is not None and f.stat().st_size > max_bytes:
                files[rel] = f"<skipped: {f.stat().st_size} bytes, over the inline cap>"
                skipped += 1
                continue
            files[rel] = inline_content if inline_content is not None else f.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            if preserve_all:
                raise
            files[rel] = f"<could not read: {exc}>"
            skipped += 1
    return {"files": files, "media": media, "media_files": media_files, "skipped": skipped}


def artifact_to_numpy(artifact: Any) -> "Any | None":
    """Convert *artifact* to a numpy array, or return None if not possible.

    Handles: torch.Tensor, list[torch.Tensor] (e.g. per-layer attentions),
    and numpy arrays.  A list of tensors is stacked along a new first axis so
    that ``attentions`` (list of ``(heads, seq, seq)``) becomes
    ``(layers, heads, seq, seq)`` — a single array that retains all the data.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    if hasattr(artifact, "detach"):  # torch.Tensor
        return artifact.detach().cpu().float().numpy()
    if isinstance(artifact, np.ndarray):
        return artifact
    if isinstance(artifact, list) and artifact and hasattr(artifact[0], "detach"):
        try:
            import torch
            return torch.stack(artifact).detach().cpu().float().numpy()
        except Exception:  # noqa: BLE001
            return None
    return None


def save_artifact_figure(artifact_dir: Path, stem: str, arr: Any) -> None:
    """Save a matplotlib figure of *arr* when the shape and stem are recognised.

    Dispatch table (first match wins):
    - 4-D + ``attn`` in stem → mean over (layers, heads) → 2-D heatmap
    - 3-D + ``attn`` in stem → mean over heads → 2-D heatmap
    - 2-D + heatmap keyword  → direct heatmap (viridis)
    - 1-D + curve keyword    → line plot
    Skips silently when matplotlib is unavailable or the shape is unrecognised.
    """
    try:
        import matplotlib.pyplot as plt
        plt.ioff()
    except ImportError:
        return

    key = stem.lower()
    # Skip logit arrays — (seq, vocab) shape is too large for a useful figure
    if "logit" in key:
        return

    _is_attn = any(k in key for k in ("attn", "attention"))

    fig = None
    try:
        ndim = arr.ndim
        if ndim == 4 and _is_attn:
            mat = arr.mean(axis=(0, 1))  # (layers, heads, seq, seq) → (seq, seq)
            n_layers, n_heads = arr.shape[0], arr.shape[1]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_layers}L × {n_heads}H)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 3 and _is_attn:
            mat = arr.mean(axis=0)  # (heads, seq, seq) → (seq, seq)
            n_heads = arr.shape[0]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0)
            ax.set_title(f"{stem}  (mean over {n_heads} heads)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and "diff" in key:
            # Signed difference map (e.g. FAIL-mean minus PASS-mean attention):
            # diverging colormap with symmetric limits so the sign is readable.
            bound = float(max(abs(arr.min()), abs(arr.max()))) or 1.0
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=-bound, vmax=bound)
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 2 and (_is_attn or any(k in key for k in ("rollout", "spatial", "map"))):
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(arr, cmap="viridis", aspect="auto")
            ax.set_title(stem)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        elif ndim == 1 and any(k in key for k in ("entropy", "score", "prob", "weight", "rollout")):
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(arr)
            ax.set_xlabel("position")
            ax.set_ylabel(stem)
            ax.set_title(stem)
            plt.tight_layout()

        if fig is not None:
            fig.savefig(artifact_dir / f"{stem}.png", dpi=100, bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    finally:
        if fig is not None:
            plt.close(fig)
