"""image_ocr / image_detect — engine-injected (no easyocr/torch needed here)."""

from __future__ import annotations

import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from evalrx.core.tool import ToolCall, ToolResult  # noqa: E402
from evalrx.models.agent import ToolExecutor  # noqa: E402
from evalrx.models.tools import detect_tool, ocr_tool  # noqa: E402


def _img(w=200, h=100):
    return Image.new("RGB", (w, h), "white")


# ----------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------
def _fake_ocr_engine(pil_image):
    # easyocr shape: (4-point polygon in px, text, confidence)
    return [([(10, 10), (60, 10), (60, 30), (10, 30)], "EXIT", 0.97)]


def test_ocr_reports_text_with_normalized_locations():
    res = ToolExecutor([ocr_tool(_img(), engine=_fake_ocr_engine)]).execute(
        ToolCall(name="image_ocr", args={})
    )
    assert isinstance(res, ToolResult)
    assert "'EXIT'" in res.text
    assert "[0.05, 0.10, 0.30, 0.30]" in res.text  # px / (200, 100)
    assert res.meta["n_detections"] == 1 and res.meta["region"] == "full"
    assert res.images == []  # OCR returns text, not images


def test_ocr_region_offsets_back_to_full_image_coords():
    def engine(region):
        assert region.size == (100, 100)  # right half of 200x100? no: bbox below
        return [([(0, 0), (10, 0), (10, 10), (0, 10)], "x", 0.5)]

    res = ToolExecutor([ocr_tool(_img(200, 100), engine=engine)]).execute(
        ToolCall(name="image_ocr", args={"bbox": [0.5, 0.0, 1.0, 1.0]})
    )
    # region starts at px x=100 -> min x fraction is 100/200 = 0.5
    assert "[0.50, 0.00," in res.text
    assert res.meta["region"] == "bbox" and res.meta["bbox_frac"] == [0.5, 0.0, 1.0, 1.0]


def test_ocr_empty_result_is_explicit():
    res = ToolExecutor([ocr_tool(_img(), engine=lambda im: [])]).execute(
        ToolCall(name="image_ocr", args={})
    )
    assert "No text detected" in res.text


# ----------------------------------------------------------------------
# Detect
# ----------------------------------------------------------------------
def _fake_detect_engine(pil_image, query):
    return [
        {"label": "dog", "score": 0.81, "box_px": [20.0, 10.0, 120.0, 90.0]},
        {"label": "dog", "score": 0.44, "box_px": [150.0, 40.0, 190.0, 95.0]},
    ]


def test_detect_reports_fractional_boxes_and_annotated_image():
    res = ToolExecutor([detect_tool(_img(), engine=_fake_detect_engine)]).execute(
        ToolCall(name="image_detect", args={"query": "dog"})
    )
    assert isinstance(res, ToolResult)
    assert "Detected 2 instance(s) of 'dog'" in res.text
    assert "[0.10, 0.10, 0.60, 0.90]" in res.text
    assert len(res.images) == 1  # annotated copy re-injected for the model
    dets = res.meta["detections"]
    assert dets[0]["score"] >= dets[1]["score"]  # sorted by confidence
    assert json.loads(json.dumps(res.meta))  # meta stays JSON-safe


def test_detect_no_hits_reports_threshold_miss():
    res = ToolExecutor([detect_tool(_img(), engine=lambda im, q: [])]).execute(
        ToolCall(name="image_detect", args={"query": "unicorn"})
    )
    assert "No instances of 'unicorn'" in res.text
    assert res.images == []


def test_detect_empty_query_uses_error_envelope():
    out = ToolExecutor([detect_tool(_img(), engine=_fake_detect_engine)]).execute(
        ToolCall(name="image_detect", args={"query": "  "})
    )
    assert isinstance(out, str) and out.startswith("[tool error in 'image_detect'")


def test_detect_without_annotation_returns_no_images():
    res = ToolExecutor([detect_tool(_img(), engine=_fake_detect_engine, annotate=False)]).execute(
        ToolCall(name="image_detect", args={"query": "dog"})
    )
    assert res.images == []
