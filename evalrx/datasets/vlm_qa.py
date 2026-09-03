"""VLM (image + text) QA loaders → CaseBatch.

``VLMQADataset`` is the generic image+text QA loader. ``TextVQASizeDataset``
wraps TextVQA-style records with answer bounding boxes into the small/medium/
large partitions used by Zhang et al. (ICLR 2025, arXiv:2502.17422).
``Spatial457Dataset`` wraps the **RyanWW/Spatial457** benchmark (6D
spatial-reasoning VQA) via the HuggingFace ``datasets`` library (optional dep —
``pip install evalrx[data]``).

The image goes into ``Inputs.image`` (kept OUT of metadata so heavy PIL objects
aren't duplicated); the gold answer is ``expected``; cases are tagged ``vlm_qa``.

Spatial457:
  Paper: "Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large
         Multimodal Models" — Wang et al., CVPR 2025 — arXiv:2502.08636
  Data:  https://huggingface.co/datasets/RyanWW/Spatial457
  Code:  https://github.com/XingruiWang/Spatial457
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evalrx.core.case import CaseBatch, FailureCase, Inputs, Provenance, Source
from evalrx.datasets.base import Dataset, read_jsonl


def _vlm_cases(
    records: Iterable[dict],
    *,
    prompt_key: str,
    answer_key: str,
    image_key: str,
    tags: set[str],
    meta_keys: tuple[str, ...] = (),
    base_meta: dict | None = None,
) -> CaseBatch:
    """Build image+text cases — image -> Inputs.image (NOT metadata); selected keys -> metadata."""
    out = CaseBatch()
    for rec in records:
        meta = dict(base_meta or {})
        for k in meta_keys:
            if k in rec:
                meta[k] = rec[k]
        out.append(FailureCase(
            inputs=Inputs(prompt=str(rec.get(prompt_key, "")), image=rec.get(image_key)),
            expected=rec.get(answer_key),
            tags=set(tags),
            provenance=Provenance(source=Source.DATASET),
            metadata=meta,
        ))
    return out


_VLM_SAMPLE = [
    {"question": "What is shown in the image?", "answer": "a placeholder", "image": None, "note": "attach a real image"},
]


class VLMQADataset(Dataset):
    """Generic image+text question-answering loader."""

    def __init__(
        self,
        records: list[dict] | None = None,
        path: str | None = None,
        *,
        prompt_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "image",
        meta_keys: tuple[str, ...] = (),
    ) -> None:
        self._records = records
        self._path = path
        self._prompt_key, self._answer_key, self._image_key = prompt_key, answer_key, image_key
        self._meta_keys = meta_keys

    @classmethod
    def from_records(cls, records: list[dict], **kw) -> "VLMQADataset":
        return cls(records=records, **kw)

    @classmethod
    def from_jsonl(cls, path: str, **kw) -> "VLMQADataset":
        return cls(path=path, **kw)

    @classmethod
    def sample(cls) -> "VLMQADataset":
        return cls(records=_VLM_SAMPLE, meta_keys=("note",))

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            read_jsonl(self._path) if self._path else _VLM_SAMPLE
        )
        return _vlm_cases(records, prompt_key=self._prompt_key, answer_key=self._answer_key,
                          image_key=self._image_key, tags={"vlm_qa"}, meta_keys=self._meta_keys)


# TextVQA size-sensitivity split from arXiv:2502.17422:
# S = answer_bbox_area / image_area; small < 0.005, medium [0.005, 0.05), large >= 0.05.
_TEXTVQA_SIZE_THRESHOLDS = {
    "small": (None, 0.005),
    "medium": (0.005, 0.05),
    "large": (0.05, None),
}
_TEXTVQA_BBOX_KEYS = (
    "answer_bbox",
    "answer_bboxes",
    "answer_box",
    "answer_boxes",
    "bbox",
    "bboxes",
    "bounding_box",
    "bounding_boxes",
    "ocr_bbox",
    "ocr_bboxes",
)


def _read_json_or_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "annotations", "questions", "records"):
            if isinstance(data.get(key), list):
                return [r for r in data[key] if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    raise ValueError(f"unsupported TextVQA annotation shape in {p}")


def _first_present(rec: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in rec and rec[key] not in (None, ""):
            return rec[key]
    return None


def _textvqa_answers(rec: dict) -> list[str]:
    raw = _first_present(rec, ("answers", "valid_answers", "answer", "gold_answer"))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        value = str(_first_present(raw, ("answer", "text", "word", "value")) or "").strip()
        return [value] if value else []
    out: list[str] = []
    if isinstance(raw, Iterable):
        for item in raw:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = str(_first_present(item, ("answer", "text", "word", "value")) or "")
            else:
                value = str(item)
            value = value.strip()
            if value and value not in out:
                out.append(value)
    return out


def _textvqa_rubric(answers: list[str]) -> str | dict:
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]
    return {"any_of": answers}


def _image_value(rec: dict, image_dir: str | Path | None, image_key: str) -> Any:
    image = rec.get(image_key)
    if image is None:
        image = _first_present(rec, ("image_path", "image_file", "file_name", "filename"))
    if isinstance(image, str) and image_dir and not image.startswith(("http://", "https://")):
        p = Path(image)
        if not p.is_absolute():
            image = str(Path(image_dir) / p)
    return image


def _image_size(rec: dict, image: Any) -> tuple[int, int] | None:
    width = _first_present(rec, ("image_width", "width", "w"))
    height = _first_present(rec, ("image_height", "height", "h"))
    if width and height:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    try:
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.size
        if isinstance(image, str) and image and Path(image).exists():
            with Image.open(image) as img:
                return img.size
    except Exception:
        return None
    return None


def _is_box_like(value: Any) -> bool:
    if isinstance(value, dict):
        keys = set(value)
        return bool({"x", "left", "x1"} & keys) and bool({"y", "top", "y1"} & keys)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            [float(v) for v in value]
        except (TypeError, ValueError):
            return False
        return True
    return False


def _flatten_boxes(value: Any) -> list[Any]:
    if value is None:
        return []
    if _is_box_like(value):
        return [value]
    if isinstance(value, dict):
        nested = _first_present(value, _TEXTVQA_BBOX_KEYS)
        return _flatten_boxes(nested)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_boxes(item))
        return out
    return []


def _bbox_from_ocr_tokens(rec: dict, answers: list[str]) -> list[Any]:
    tokens = _first_present(rec, ("ocr_info", "ocr_tokens", "ocr", "ocr_words"))
    if not isinstance(tokens, Iterable) or isinstance(tokens, (str, bytes)):
        return []
    answer_terms = {a.strip().lower() for a in answers if str(a).strip()}
    answer_words = {
        part
        for answer in answer_terms
        for part in str(answer).replace("/", " ").replace("-", " ").split()
        if part
    }
    out: list[Any] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = str(_first_present(token, ("text", "word", "ocr_text", "value")) or "").strip().lower()
        if answer_terms and text not in answer_terms and text not in answer_words:
            continue
        out.extend(_flatten_boxes(_first_present(token, _TEXTVQA_BBOX_KEYS)))
    return out


def _box_to_norm_xyxy(
    box: Any,
    *,
    image_size: tuple[int, int],
    bbox_format: str,
) -> tuple[float, float, float, float, bool] | None:
    width, height = image_size
    fmt = bbox_format.lower()
    if isinstance(box, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(box):
            vals = [box["x1"], box["y1"], box["x2"], box["y2"]]
            fmt = "xyxy"
        elif {"left", "top", "right", "bottom"}.issubset(box):
            vals = [box["left"], box["top"], box["right"], box["bottom"]]
            fmt = "xyxy"
        elif {"x", "y", "w", "h"}.issubset(box):
            vals = [box["x"], box["y"], box["w"], box["h"]]
            fmt = "xywh"
        elif {"x", "y", "width", "height"}.issubset(box):
            vals = [box["x"], box["y"], box["width"], box["height"]]
            fmt = "xywh"
        else:
            return None
    else:
        try:
            vals = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        if fmt == "auto":
            if all(0.0 <= v <= 1.0 for v in vals) and vals[2] > vals[0] and vals[3] > vals[1]:
                fmt = "xyxy"
            else:
                fmt = "xywh"

    try:
        x0, y0, a, b = [float(v) for v in vals]
    except (TypeError, ValueError):
        return None
    normalized = all(0.0 <= v <= 1.0 for v in (x0, y0, a, b))
    if fmt == "xyxy":
        x1, y1, x2, y2 = x0, y0, a, b
    else:
        x1, y1, x2, y2 = x0, y0, x0 + a, y0 + b
    if not normalized:
        x1, x2 = x1 / max(1, width), x2 / max(1, width)
        y1, y2 = y1 / max(1, height), y2 / max(1, height)
    left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    degenerate = right <= left or bottom <= top
    if right <= left:
        right = min(1.0, left + 1.0 / max(1, width))
        if right <= left:
            left = max(0.0, right - 1.0 / max(1, width))
    if bottom <= top:
        bottom = min(1.0, top + 1.0 / max(1, height))
        if bottom <= top:
            top = max(0.0, bottom - 1.0 / max(1, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom, degenerate


def _union_norm_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    return left, top, right, bottom


def _size_split_for_ratio(ratio: float) -> str:
    if ratio < 0.005:
        return "small"
    if ratio < 0.05:
        return "medium"
    return "large"


def _split_matches(ratio: float, size_split: str) -> bool:
    if size_split == "all":
        return True
    lo, hi = _TEXTVQA_SIZE_THRESHOLDS[size_split]
    return (lo is None or ratio >= lo) and (hi is None or ratio < hi)


class TextVQASizeDataset(Dataset):
    """TextVQA-style VQA records partitioned by answer-bbox relative size.

    This loader expects local records with a question, one or more accepted
    answers, an image path/object, and an answer bounding box. It supports both
    JSONL and JSON files. The split follows arXiv:2502.17422:
    ``small`` ``S < 0.005``, ``medium`` ``0.005 <= S < 0.05``, and ``large``
    ``S >= 0.05``, where ``S`` is answer bbox area divided by image area.
    """

    SPLITS = ("small", "medium", "large", "all")

    def __init__(
        self,
        records: list[dict] | None = None,
        path: str | None = None,
        *,
        image_dir: str | Path | None = None,
        size_split: str = "small",
        max_samples: int | None = None,
        bbox_format: str = "auto",
        question_key: str = "question",
        image_key: str = "image",
    ) -> None:
        if size_split not in self.SPLITS:
            raise ValueError(f"unknown TextVQA size_split {size_split!r}; choose from {self.SPLITS}")
        if bbox_format not in {"auto", "xywh", "xyxy"}:
            raise ValueError("bbox_format must be 'auto', 'xywh', or 'xyxy'")
        self._records = records
        self._path = path
        self.image_dir = image_dir
        self.size_split = size_split
        self.max_samples = max_samples
        self.bbox_format = bbox_format
        self.question_key = question_key
        self.image_key = image_key

    @classmethod
    def from_records(cls, records: list[dict], **kw) -> "TextVQASizeDataset":
        return cls(records=records, **kw)

    @classmethod
    def from_jsonl(cls, path: str, **kw) -> "TextVQASizeDataset":
        return cls(path=path, **kw)

    @classmethod
    def sample(cls) -> "TextVQASizeDataset":
        return cls(records=[
            {
                "question_id": "textvqa_size_sample_0",
                "image": None,
                "image_width": 1000,
                "image_height": 1000,
                "question": "What word is printed on the small sign?",
                "answers": ["moma", "MoMA"],
                "answer_bbox": [100, 120, 40, 12],
            },
        ])

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            _read_json_or_jsonl(self._path) if self._path else self.sample()._records
        )
        cases = CaseBatch()
        for idx, rec in enumerate(records or []):
            answers = _textvqa_answers(rec)
            image = _image_value(rec, self.image_dir, self.image_key)
            image_size = _image_size(rec, image)
            if image_size is None:
                continue
            raw_boxes = _flatten_boxes(_first_present(rec, _TEXTVQA_BBOX_KEYS))
            if not raw_boxes:
                raw_boxes = _bbox_from_ocr_tokens(rec, answers)
            parsed_boxes = []
            for raw in raw_boxes:
                parsed = _box_to_norm_xyxy(
                    raw,
                    image_size=image_size,
                    bbox_format=str(rec.get("bbox_format") or self.bbox_format),
                )
                if parsed is not None:
                    parsed_boxes.append(parsed)
            norm_boxes = [parsed[:4] for parsed in parsed_boxes]
            if not norm_boxes:
                continue
            bbox = _union_norm_boxes(norm_boxes)
            bbox_degenerate = any(parsed[-1] for parsed in parsed_boxes)
            ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if not _split_matches(ratio, self.size_split):
                continue
            question_id = str(_first_present(rec, ("question_id", "id", "qid")) or f"textvqa_{idx}")
            image_id = _first_present(rec, ("image_id", "image_name", "image_file", "filename"))
            metadata = {
                "dataset": "textvqa",
                "question_id": question_id,
                "image_id": image_id,
                "size_split": _size_split_for_ratio(ratio),
                "requested_size_split": self.size_split,
                "answer_bbox_xyxy_norm": [round(v, 6) for v in bbox],
                "bbox_area_ratio": ratio,
                "bbox_degenerate": bbox_degenerate,
                "answers": answers,
            }
            cases.append(FailureCase(
                id=question_id,
                inputs=Inputs(prompt=str(rec.get(self.question_key, "")), image=image),
                expected=_textvqa_rubric(answers),
                tags={"vlm_qa", "textvqa", f"textvqa_{metadata['size_split']}"},
                provenance=Provenance(source=Source.DATASET),
                metadata=metadata,
            ))
            if self.max_samples is not None and len(cases) >= self.max_samples:
                break
        return cases


# Spatial457 — RyanWW/Spatial457 row schema: {image: PIL, image_filename, question, answer, program, question_index}
_SPATIAL457_SAMPLE = [
    {"image": None, "image_filename": "superCLEVR_new_000001.png", "question_index": 100001,
     "question": "Is the large red object in front of the yellow car?", "answer": "True"},
    {"image": None, "image_filename": "superCLEVR_new_000002.png", "question_index": 100002,
     "question": "How many objects are behind the blue cube?", "answer": "2"},
]


class Spatial457Dataset(Dataset):
    """RyanWW/Spatial457 — 6D spatial-reasoning VQA (CVPR'25), 7 cascading subtypes.

    ``subset`` selects a difficulty/question-type split.  ``load()`` pulls from
    HuggingFace (``pip install evalrx[data]``); use ``from_records`` / ``sample``
    offline.  Maps each row to a ``vlm_qa`` FailureCase (image -> Inputs.image,
    answer -> expected; image_filename/question_index/subset -> metadata).
    """

    SUBTYPES = (
        "L1_single", "L2_objects", "L3_2d_spatial", "L4_occ", "L4_pose",
        "L5_6d_spatial", "L5_collision",
    )

    def __init__(
        self,
        subset: str = "L5_6d_spatial",
        split: str = "test",
        *,
        records: list[dict] | None = None,
        hf_repo: str = "RyanWW/Spatial457",
        max_samples: int | None = None,
    ) -> None:
        if subset not in self.SUBTYPES:
            raise ValueError(f"unknown Spatial457 subset {subset!r}; choose from {self.SUBTYPES}")
        self.subset = subset
        self.split = split
        self.hf_repo = hf_repo
        self.max_samples = max_samples
        self._records = records

    @classmethod
    def from_records(cls, records: list[dict], subset: str = "L5_6d_spatial") -> "Spatial457Dataset":
        return cls(subset=subset, records=records)

    @classmethod
    def sample(cls, subset: str = "L5_6d_spatial") -> "Spatial457Dataset":
        return cls(subset=subset, records=_SPATIAL457_SAMPLE)

    def _load_hf(self) -> Iterable[dict]:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Spatial457Dataset.load() needs the HuggingFace 'datasets' library — "
                "pip install evalrx[data] (or use from_records()/sample() offline)."
            ) from e
        try:  # subtypes are exposed as dataset configs
            ds = load_dataset(self.hf_repo, self.subset, split=self.split)
        except Exception:  # fall back: single config, filter by a subtype column if present
            ds = load_dataset(self.hf_repo, split=self.split)
            for col in ("question_type", "subtype", "level"):
                if col in (getattr(ds, "column_names", None) or []):
                    ds = ds.filter(lambda r, c=col: r.get(c) == self.subset)
                    break
        if self.max_samples is not None:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        return ds

    def load(self) -> CaseBatch:
        rows = self._records if self._records is not None else self._load_hf()
        return _vlm_cases(
            rows, prompt_key="question", answer_key="answer", image_key="image",
            tags={"vlm_qa", "spatial457", self.subset},
            meta_keys=("image_filename", "question_index", "program"),
            base_meta={"dataset": "spatial457", "subset": self.subset},
        )


# VQA-RAD — flaviagiammarino/vqa-rad row schema: {image: PIL, question, answer}
_VQA_RAD_SAMPLE = [
    {"image": None, "question": "what imaging modality was used?", "answer": "ct"},
    {"image": None, "question": "is there evidence of a pneumothorax?", "answer": "no"},
    {"image": None, "question": "is the heart enlarged?", "answer": "yes"},
    {"image": None, "question": "what plane is this image taken in?", "answer": "axial"},
]

# Question fragments that identify "easy" identification questions (modality /
# plane / organ) a general VLM reliably answers — the M5 control (PASS) group.
_VQA_RAD_EASY_FRAGMENTS = (
    "modality", "what plane", "which plane", "plane is", "what organ",
    "which organ", "organ system", "part of the body", "what type of imaging",
    "what kind of image", "what imaging", "mri or ct", "ct or mri",
)


def _categorize_vqa_rad(question: str, answer: str) -> str:
    """``"easy"`` (modality/plane/organ), ``"presence"`` (closed yes/no), or ``"other"``."""
    q = " ".join(str(question).lower().split())
    if any(frag in q for frag in _VQA_RAD_EASY_FRAGMENTS):
        return "easy"
    if str(answer).strip().lower() in {"yes", "no"}:
        return "presence"
    return "other"


def _easy_answer_rubric(gold: str) -> dict:
    """Token-level ``any_of`` rubric for open identification answers.

    VQA-RAD gold strings are messy ("xray - plain film", tab-separated organ
    lists) — whole-string matching would mislabel correct answers as FAIL and
    pollute the control group.  Accept any significant gold token instead, with
    a cheap plural tolerance (kidneys → kidney).
    """
    import re as _re

    tokens: list[str] = []
    for tok in _re.findall(r"[a-z0-9]+", str(gold).lower()):
        if len(tok) < 2 or tok in tokens:
            continue
        tokens.append(tok)
        if tok.endswith("s") and len(tok) > 3 and tok[:-1] not in tokens:
            tokens.append(tok[:-1])
    return {"any_of": tokens or [str(gold).strip().lower()]}


class VQARADDataset(Dataset):
    """VQA-RAD — radiology VQA (Lau et al., 2018), public domain (CC0).

    Builds a **diagnosis-ready** case mix for the VL failure-analysis loop:

    - ``n_easy`` identification questions (modality/plane/organ) the model
      reliably PASSES — M5's control group;
    - ``n_presence`` closed yes/no finding-presence questions, balanced between
      gold "yes" and gold "no" — where presence hallucination (yes-bias)
      concentrates the failures.

    Presence cases carry ``metadata["pope_label"]`` so the ``pope`` analyzer can
    score them and emit per-case ``false_positive`` / ``false_negative``
    mechanism signals, plus a strict ``{"all_of": [gold], "none_of": [other]}``
    rubric for the discovery scorer.  Easy cases get a token-level ``any_of``
    rubric (gold strings are messy free text).

    Data: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
    Paper: "A dataset of clinically generated visual questions and answers
           about radiology images" — Lau et al., Scientific Data 2018.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        records: list[dict] | None = None,
        hf_repo: str = "flaviagiammarino/vqa-rad",
        n_easy: int = 6,
        n_presence: int = 12,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.hf_repo = hf_repo
        self.n_easy = n_easy
        self.n_presence = n_presence
        self.seed = seed
        self._records = records

    @classmethod
    def from_records(cls, records: list[dict], **kw) -> "VQARADDataset":
        return cls(records=records, **kw)

    @classmethod
    def sample(cls) -> "VQARADDataset":
        return cls(records=_VQA_RAD_SAMPLE, n_easy=2, n_presence=2)

    def _load_hf(self) -> Iterable[dict]:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "VQARADDataset.load() needs the HuggingFace 'datasets' library — "
                "pip install evalrx[data] (or use from_records()/sample() offline)."
            ) from e
        return load_dataset(self.hf_repo, split=self.split)

    def load(self) -> CaseBatch:
        import random

        rows = self._records if self._records is not None else self._load_hf()

        easy: list[dict] = []
        pres_yes: list[dict] = []
        pres_no: list[dict] = []
        for rec in rows:
            cat = _categorize_vqa_rad(rec.get("question", ""), rec.get("answer", ""))
            if cat == "easy":
                easy.append(rec)
            elif cat == "presence":
                (pres_yes if str(rec["answer"]).strip().lower() == "yes" else pres_no).append(rec)

        rng = random.Random(self.seed)
        rng.shuffle(easy)
        rng.shuffle(pres_yes)
        rng.shuffle(pres_no)

        half = self.n_presence // 2
        picked_pres = pres_yes[:half] + pres_no[: self.n_presence - half]
        picked_easy = easy[: self.n_easy]

        out = CaseBatch()
        for i, rec in enumerate(picked_easy):
            out.append(FailureCase(
                id=f"rad_easy_{i}",
                inputs=Inputs(
                    prompt=f"{str(rec['question']).strip()} Answer briefly.",
                    image=rec.get("image"),
                ),
                expected=_easy_answer_rubric(rec["answer"]),
                tags={"vlm_qa", "med_vqa", "vqa_rad", "easy"},
                provenance=Provenance(source=Source.DATASET),
                metadata={"dataset": "vqa_rad", "category": "easy",
                          "gold_answer": str(rec["answer"]).strip()},
            ))
        for i, rec in enumerate(picked_pres):
            gold = str(rec["answer"]).strip().lower()
            other = "no" if gold == "yes" else "yes"
            out.append(FailureCase(
                id=f"rad_pres_{i}",
                inputs=Inputs(
                    prompt=f"{str(rec['question']).strip()} Answer yes or no.",
                    image=rec.get("image"),
                ),
                expected={"all_of": [gold], "none_of": [other]},
                tags={"vlm_qa", "med_vqa", "vqa_rad", "presence"},
                provenance=Provenance(source=Source.DATASET),
                metadata={"dataset": "vqa_rad", "category": "presence",
                          "pope_label": gold, "gold_answer": gold},
            ))
        return out
