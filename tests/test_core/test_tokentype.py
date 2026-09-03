"""TokenTypeMap builder — pure logic, synthetic inputs (no torch / no model)."""

from __future__ import annotations

from evalrx.core.spec import VisionSpec
from evalrx.core.tokentype import TokenTypeMap, build_token_type_map


class _Cfg:
    """Minimal stand-in for a model config with a nested vision_config."""

    def __init__(self, image_token_id=999, merge=2):
        self.image_token_id = image_token_id
        self.vision_config = type("V", (), {"spatial_merge_size": merge})()


def test_grid_thw_id_match_single_image():
    # 4 image tokens (id 999) between text; grid 2x4 pre-merge, merge=2 -> post 1x2 => 2 tokens.
    # Use a post-merge grid of 2x2 = 4 to match the 4 image tokens.
    spec = VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw")
    ids = [1, 2, 999, 999, 999, 999, 3]                 # 4 image tokens at positions 2..5
    enc = {"image_grid_thw": [[1, 4, 4]]}               # pre-merge (t,h,w); /2 -> (1,2,2)=4 tokens
    ttm = build_token_type_map(ids, enc, _Cfg(merge=2), spec)
    assert isinstance(ttm, TokenTypeMap)
    assert ttm.image_pos == [2, 3, 4, 5]
    assert ttm.text_pos == [0, 1, 6]
    assert ttm.grids == [(1, 2, 2)]
    assert ttm.n_images == 1 and ttm.has_image
    # row-major mapping over the 2x2 post-merge grid
    assert ttm.patches == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]
    assert ttm.image_token_id == 999


def test_prefers_mm_token_type_ids_when_present():
    spec = VisionSpec(grid_source="grid_thw")  # prefer_mm_token_type_ids defaults True
    ids = [5, 5, 5, 5]                          # id-match would find nothing (no 999)
    enc = {"mm_token_type_ids": [0, 1, 1, 0], "image_grid_thw": [[1, 2, 2]]}
    ttm = build_token_type_map(ids, enc, _Cfg(), spec)
    assert ttm.image_pos == [1, 2]              # from mm_token_type_ids, not id-match


def test_grid_hw_kimi_style_halves():
    spec = VisionSpec(image_token_id_attr="image_token_id", grid_source="grid_hw",
                      merge_size_attr="vision_config.spatial_merge_size")
    ids = [1, 999, 999, 999, 999]
    enc = {"image_grid_hws": [[4, 4]]}          # pre-merge 4x4; /2 -> 2x2 = 4 tokens
    ttm = build_token_type_map(ids, enc, _Cfg(merge=2), spec)
    assert ttm.grids == [(1, 2, 2)]
    assert ttm.image_pos == [1, 2, 3, 4]


def test_no_image_is_empty_map():
    spec = VisionSpec(image_token_id_attr="image_token_id", grid_source="grid_thw")
    ttm = build_token_type_map([1, 2, 3], {}, _Cfg(), spec)
    assert not ttm.has_image and ttm.image_pos == [] and ttm.grids == []
    assert ttm.text_pos == [0, 1, 2]


def test_fixed_token_budget_gemma_style():
    spec = VisionSpec(image_token_id_attr="image_token_id", grid_source="fixed", fixed_tokens_per_tile=4)
    ids = [1, 999, 999, 999, 999, 2]            # 4 image tokens = 1 tile of 4 (2x2)
    ttm = build_token_type_map(ids, {}, _Cfg(), spec)
    assert ttm.grids == [(1, 2, 2)]
    assert len(ttm.patches) == 4
