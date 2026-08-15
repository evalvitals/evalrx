"""Visual tools — zoom_in: coordinate modes, guards, artifacts, error envelope."""

from __future__ import annotations

import json
import os

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from evalvitals.core.tool import ToolCall, ToolResult  # noqa: E402
from evalvitals.models.agent import ToolExecutor  # noqa: E402
from evalvitals.models.tools import zoom_in_tool  # noqa: E402


def _img(w=640, h=480):
    return Image.new("RGB", (w, h), "white")


def _run(tool, **args):
    return ToolExecutor([tool]).execute(ToolCall(name="image_zoom_in", args=args))


def test_normalized_bbox_crops_the_right_pixels():
    res = _run(zoom_in_tool(_img()), bbox=[0.5, 0.0, 1.0, 1.0])
    assert isinstance(res, ToolResult)
    assert res.meta["coord_mode"] == "normalized"
    assert res.meta["crop_px"] == [320, 0, 640, 480]
    assert len(res.images) == 1


def test_pixel_coordinates_are_autodetected():
    res = _run(zoom_in_tool(_img()), bbox=[320, 0, 640, 480])
    assert res.meta["coord_mode"] == "pixel"
    assert res.meta["bbox_frac"] == [0.5, 0.0, 1.0, 1.0]


def test_qwen_grid_1000_coordinates_are_autodetected():
    # 750 exceeds the 480-px height -> cannot be pixels -> 0-1000 grid
    res = _run(zoom_in_tool(_img()), bbox=[250, 250, 750, 750])
    assert res.meta["coord_mode"] == "grid_1000"
    assert res.meta["bbox_frac"] == [0.25, 0.25, 0.75, 0.75]


def test_swapped_and_out_of_range_corners_are_sanitized():
    res = _run(zoom_in_tool(_img()), bbox=[1.2, 0.9, 0.5, 0.1])
    x0, y0, x1, y1 = res.meta["bbox_frac"]
    assert x0 <= x1 and y0 <= y1
    assert 0.0 <= x0 and x1 <= 1.0 and 0.0 <= y0 and y1 <= 1.0


def test_sliver_bbox_is_expanded_to_minimum_side():
    res = _run(zoom_in_tool(_img()), bbox=[0.5, 0.5, 0.5, 0.5])
    x0, y0, x1, y1 = res.meta["bbox_frac"]
    assert x1 - x0 >= 0.05 and y1 - y0 >= 0.05


def test_crop_is_upscaled_but_capped():
    res = _run(zoom_in_tool(_img()), bbox=[0.0, 0.0, 0.1, 0.1])  # 64x48 crop, 4x cap binds
    w, h = res.images[0].size
    assert (w, h) == (256, 192)


def test_save_dir_writes_numbered_crops(tmp_path):
    tool = zoom_in_tool(_img(), save_dir=str(tmp_path))
    first = _run(tool, bbox=[0.0, 0.0, 0.5, 0.5])
    second = _run(tool, bbox=[0.5, 0.5, 1.0, 1.0])
    assert os.path.basename(first.meta["saved_path"]) == "zoom_00.png"
    assert os.path.basename(second.meta["saved_path"]) == "zoom_01.png"
    assert os.path.exists(second.meta["saved_path"])


def test_malformed_bbox_returns_standard_error_envelope():
    out = _run(zoom_in_tool(_img()), bbox=[0.1, 0.2])
    assert isinstance(out, str) and out.startswith("[tool error in 'image_zoom_in'")


def test_tool_accepts_a_path(tmp_path):
    p = tmp_path / "x.png"
    _img(100, 100).save(p)
    res = _run(zoom_in_tool(str(p)), bbox=[0.0, 0.0, 1.0, 1.0])
    assert res.meta["crop_px"] == [0, 0, 100, 100]


def test_meta_is_json_serializable():
    res = _run(zoom_in_tool(_img()), bbox=[0.0, 0.0, 0.5, 0.5], target="the mug")
    assert json.loads(json.dumps(res.meta))["target"] == "the mug"
