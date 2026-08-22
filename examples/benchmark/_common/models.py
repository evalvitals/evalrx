"""The benchmark support matrix: which spec key a (size, modality, backend) cell loads.

The docker image is keyed by the FAMILY (its runtime stack); the size is a
runtime argument and the modality/dataset is another. ``resolve()`` turns the
``--model <size>`` / ``--modality`` pair into the registered spec key:

* Qwen3.5 text specs load the language tower only (``qwen3.5-2b``), the
  vision-tower variants are the ``-vl`` keys; the audio cell is Qwen3-Omni.
* Gemma 4 is one spec per checkpoint for every modality (natively omni).
* Nemotron 3 Nano: hf_local loads the BF16 checkpoints; the ModelOpt FP8
  checkpoints the benchmark was specified with are vLLM-only on Ampere, so they
  are the ``backend="endpoint"`` resolution of the same cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODALITIES = ("vlm", "llm", "alm")


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    docker_target: str            # stage name in examples/benchmark/docker/Dockerfile
    image: str                    # image tag the leaf compose files reference


@dataclass(frozen=True)
class Size:
    key: str                      # --model value
    family: str
    label: str
    specs: dict                   # modality -> spec key for hf_local
    endpoint_specs: dict = field(default_factory=dict)   # modality -> spec key for backend=endpoint
    gpus: int = 1                 # cards the BF16 weights need (informational + device default)
    attn_impl: str = "sdpa"       # "eager" where the model code has no SDPA dispatch (remote NemotronH)
    note: str = ""

    @property
    def modalities(self) -> tuple:
        return tuple(m for m in MODALITIES if m in self.specs)

    @property
    def default_device(self) -> str:
        return "auto" if self.gpus > 1 else "cuda"


FAMILIES: dict[str, Family] = {
    "qwen": Family("qwen", "Qwen", "qwen", "evalvitals-bench-qwen"),
    "gemma": Family("gemma", "Gemma 4", "gemma", "evalvitals-bench-gemma"),
    "nemotron": Family("nemotron", "Nemotron 3 Nano", "nemotron", "evalvitals-bench-nemotron"),
}

_GEMMA = ("e2b", "e4b", "12b")

SIZES: dict[str, Size] = {
    # --- Qwen -----------------------------------------------------------------
    **{
        f"qwen3.5-{s}": Size(
            key=f"qwen3.5-{s}", family="qwen", label=f"Qwen3.5-{s.upper()}",
            specs={"llm": f"qwen3.5-{s}", "vlm": f"qwen3.5-{s}-vl"},
            note="same checkpoint: text spec loads the language tower only, -vl adds the vision tower",
        )
        for s in ("2b", "4b", "9b")
    },
    "qwen3-omni-30b-a3b": Size(
        key="qwen3-omni-30b-a3b", family="qwen", label="Qwen3-Omni-30B-A3B-Instruct",
        specs={"alm": "qwen3-omni-30b-a3b-instruct"},
        gpus=2, note="~66 GB BF16 (thinker + talker): device=auto over two 48 GB cards",
    ),
    # --- Gemma 4 --------------------------------------------------------------
    **{
        f"gemma-4-{s}": Size(
            key=f"gemma-4-{s}", family="gemma", label=f"Gemma 4 {s.upper()}-it",
            specs={m: f"gemma-4-{s}-it" for m in MODALITIES},
            note="natively text+image+audio; one spec serves every modality",
        )
        for s in _GEMMA
    },
    # --- Nemotron 3 Nano ------------------------------------------------------
    "nemotron-3-nano-4b": Size(
        key="nemotron-3-nano-4b", family="nemotron", label="Nemotron-3-Nano-4B",
        specs={"llm": "nemotron-3-nano-4b"},
        endpoint_specs={"llm": "nemotron-3-nano-4b-fp8"},
        attn_impl="eager",
        note="hf_local = BF16 checkpoint (remote NemotronH code: eager attention only); "
             "the FP8 export is the endpoint (vLLM) resolution",
    ),
    "nemotron-3-nano-omni-30b-a3b": Size(
        key="nemotron-3-nano-omni-30b-a3b", family="nemotron",
        label="Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
        specs={"vlm": "nemotron-3-nano-omni-30b-a3b-reasoning",
               "alm": "nemotron-3-nano-omni-30b-a3b-reasoning"},
        endpoint_specs={"vlm": "nemotron-3-nano-omni-30b-a3b-reasoning-fp8",
                        "alm": "nemotron-3-nano-omni-30b-a3b-reasoning-fp8"},
        gpus=2, attn_impl="eager",
        note="62 GB BF16: device=auto over two 48 GB cards; FP8 (33 GB) is vLLM-only; "
             "remote NemotronH code: eager attention only",
    ),
}


@dataclass(frozen=True)
class Resolved:
    spec_key: str
    modality: str
    backend: str
    size: Size | None             # None when --model named a raw spec key
    family: Family | None

    @property
    def label(self) -> str:
        return self.size.label if self.size is not None else self.spec_key


def resolve(model: str, modality: str, backend: str = "hf_local") -> Resolved:
    """``--model`` is a size key from :data:`SIZES`; a registered spec key is
    accepted as an escape hatch (ad-hoc comparisons against models outside the
    matrix), with no family/image bookkeeping."""
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; one of {MODALITIES}")
    if backend not in ("hf_local", "endpoint"):
        raise ValueError(f"unknown backend {backend!r}; hf_local or endpoint")
    size = SIZES.get(model)
    if size is None:
        from evalvitals.specs import REGISTRY

        if model in REGISTRY:
            return Resolved(spec_key=model, modality=modality, backend=backend, size=None, family=None)
        raise KeyError(
            f"unknown --model {model!r}. Sizes: {', '.join(SIZES)}; or any registered spec key"
        )
    table = size.endpoint_specs if backend == "endpoint" else size.specs
    key = table.get(modality) or size.specs.get(modality)
    if key is None:
        raise ValueError(
            f"{size.label} ({model}) has no {modality} cell; it runs {', '.join(size.modalities)}"
        )
    return Resolved(spec_key=key, modality=modality, backend=backend, size=size,
                    family=FAMILIES[size.family])


def cells():
    """Every (modality, family, size) cell of the matrix, in README order."""
    for modality in MODALITIES:
        for family in FAMILIES:
            for size in SIZES.values():
                if size.family == family and modality in size.specs:
                    yield modality, FAMILIES[family], size


def matrix_text() -> str:
    lines = []
    for modality in MODALITIES:
        lines.append(f"[{modality}]")
        for _m, family, size in (c for c in cells() if c[0] == modality):
            ep = size.endpoint_specs.get(modality)
            extra = f"  (endpoint: {ep})" if ep else ""
            gpu = f"  gpus={size.gpus}" if size.gpus > 1 else ""
            lines.append(f"  {family.key:9s} {size.key:32s} -> {size.specs[modality]}{extra}{gpu}")
    return "\n".join(lines)
