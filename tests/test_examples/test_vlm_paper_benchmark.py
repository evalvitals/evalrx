"""Pure contract checks for the reproducible VLM-paper benchmark example."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_runner():
    path = (
        Path(__file__).resolve().parents[2] / "examples" / "vlm_paper_benchmark" / "run_autofix.py"
    )
    spec = importlib.util.spec_from_file_location("vlm_paper_benchmark_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_downloader():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "vlm_paper_benchmark"
        / "download_benchmarks.py"
    )
    spec = importlib.util.spec_from_file_location("vlm_paper_benchmark_downloader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_vicrop():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "vlm_paper_benchmark"
        / "run_hf_vicrop.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("vlm_paper_vicrop", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_mllms_know_paper_case_and_unit_normalization():
    runner = _load_runner()

    assert "mllms_know_textvqa_small" in runner.PAPER_IDS
    assert runner.normalized("345 ml") == runner.normalized("345ml")


def test_pope_conditions_remain_separate_paper_slices():
    root = Path(__file__).resolve().parents[2] / "examples" / "vlm_paper_benchmark"
    papers = json.loads((root / "papers.json").read_text())["papers"]
    by_id = {paper["id"]: paper for paper in papers}

    assert by_id["pope"]["config"] == "Full"
    assert by_id["pope"]["split"] == "adversarial"
    assert by_id["pope_popular"]["split"] == "popular"
    assert by_id["pope_random"]["split"] == "random"


def test_paper_oracle_bbox_is_not_passed_to_autofix_cases():
    runner = _load_runner()
    row = {
        "id": "case-1",
        "question": "Read the label",
        "expected": ["yes", "yes", "yes"],
        "image": "image.png",
        "task": "vqa_consensus",
        "options": [],
        "metadata": {
            "failure_axis": "small visual detail",
            "paper_oracle_bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2],
        },
    }
    baseline = {"cases": [{"id": "case-1", "output": "no", "correct": False}]}

    case = runner.make_cases([row], baseline)[0]

    assert "paper_oracle_bbox_xyxy_norm" not in case.metadata


def test_make_cases_can_preserve_a_paper_prompt_contract():
    runner = _load_runner()
    row = {
        "id": "case-1",
        "question": "Is there a bicycle?",
        "expected": "yes",
        "image": "image.png",
        "task": "yes_no",
        "options": [],
        "metadata": {"failure_axis": "object hallucination"},
    }
    baseline = {"cases": [{"id": "case-1", "output": "no", "correct": False}]}

    case = runner.make_cases([row], baseline, prompt_fn=lambda item: item["question"])[0]

    assert case.inputs.prompt == "Is there a bicycle?"


def test_hf_runner_exposes_stable_paper_candidate_names():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "vlm_paper_benchmark"
        / "run_hf_autofix.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("vlm_paper_hf", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert "vcd_diffusion_noise" in module.PAPER_CANDIDATE_NAMES
    assert "opera_overtrust_binary" in module.PAPER_CANDIDATE_NAMES
    assert "ifcd_truthx_contrast" in module.PAPER_CANDIDATE_NAMES
    assert "vicrop_relative_attention" in module.PAPER_CANDIDATE_NAMES
    assert "vicrop_consensus_guard" in module.PAPER_CANDIDATE_NAMES
    assert "pai_image_attention" in module.PAPER_CANDIDATE_NAMES


def test_hf_runner_artifact_fingerprint_is_content_based(tmp_path):
    path = tmp_path / "truthx.pt"
    path.write_bytes(b"truthx-artifact")
    module_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "vlm_paper_benchmark"
        / "run_hf_autofix.py"
    )
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("vlm_paper_hf_artifact", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert module.artifact_sha256(str(path)) == "7de136df4bf15852d15b937690e893f6f3b8373e56f73821a578650b0d5aa2d1"
    assert module.artifact_sha256(None) is None


def test_vcd_source_pope_prompt_keeps_the_released_one_word_instruction():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "vlm_paper_benchmark"
        / "run_hf_autofix.py"
    )
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("vlm_paper_hf_prompt", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert module.source_pope_prompt({"question": "Is there a bicycle?"}) == (
        "Is there a bicycle? Please answer this question with one word."
    )
    contract, prompt = module.paper_prompt_contract(["vcd_diffusion_noise"])
    assert contract == "vcd_pope_question_plus_one_word"
    assert prompt({"question": "Is there a bicycle?"}).endswith("one word.")
    contract, prompt = module.paper_prompt_contract(["pai_image_attention"])
    assert contract == "pope_raw_question"
    assert prompt({"question": "Is there a bicycle?"}) == "Is there a bicycle?"
    contract, prompt = module.paper_prompt_contract(["opera_overtrust_binary"])
    assert contract == "pope_raw_question"
    assert prompt({"question": "Is there a bicycle?"}) == "Is there a bicycle?"
    contract, prompt = module.paper_prompt_contract(["ifcd_truthx_contrast"])
    assert contract == "pope_raw_question"
    assert prompt({"question": "Is there a bicycle?"}) == "Is there a bicycle?"


def test_paper_casebook_has_seven_mechanism_defined_cases():
    root = Path(__file__).resolve().parents[2] / "examples" / "vlm_paper_benchmark"
    casebook = json.loads((root / "paper_casebook.json").read_text())
    cases = casebook["cases"]

    assert len(cases) == 7
    for case in cases:
        assert {"paper_url", "dataset_case", "mechanism", "paper_repair", "requires"} <= set(case)
        assert case["requires"]


def test_literature_matrix_covers_eleven_source_papers():
    root = Path(__file__).resolve().parents[2] / "examples" / "vlm_paper_benchmark"
    matrix = json.loads((root / "literature_matrix.json").read_text())

    assert len(matrix["papers"]) >= 11
    assert {"vcd", "icd", "opera", "pai", "vstar", "mllms_know", "ifcd"} <= {
        paper["id"] for paper in matrix["papers"]
    }


def test_vicrop_sliding_window_prefers_relative_attention_peak():
    import numpy as np

    vicrop = _load_vicrop()
    box = vicrop.sliding_window_box(
        np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]),
        (900, 900),
        bbox_size=300,
    )

    left, top, right, bottom = box
    assert left <= 0.5 <= right
    assert top <= 0.5 <= bottom


def test_library_vicrop_crop_accepts_a_path(tmp_path):
    from PIL import Image

    from evalvitals.models.paper_methods.vicrop import crop_from_box

    path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), color="white").save(path)

    crop = crop_from_box(str(path), (0.1, 0.25, 0.6, 0.75))

    assert crop.size == (50, 40)


def test_vicrop_excludes_seen_items_by_content_not_sample_id(tmp_path):
    vicrop = _load_vicrop()
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for directory in (old_dir, new_dir):
        (directory / "images").mkdir(parents=True)
    (old_dir / "images" / "shared.png").write_bytes(b"shared-image")
    (new_dir / "images" / "shared.png").write_bytes(b"shared-image")
    (new_dir / "images" / "fresh.png").write_bytes(b"fresh-image")
    shared = {
        "id": "different-sample-position",
        "task": "vqa_consensus",
        "question": "What is shown?",
        "expected": ["label"],
        "options": [],
        "image": "images/shared.png",
        "metadata": {"paper_oracle_bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2]},
    }
    fresh = {**shared, "id": "fresh", "question": "What color?", "image": "images/fresh.png"}
    paper = "mllms_know_textvqa_small.jsonl"
    (old_dir / paper).write_text(json.dumps(shared) + "\n")
    (new_dir / paper).write_text(json.dumps(shared) + "\n" + json.dumps(fresh) + "\n")

    rows = vicrop.load_records_from(str(new_dir), "mllms_know_textvqa_small")
    unseen, excluded = vicrop.exclude_seen_rows(rows, [str(old_dir)])

    assert excluded == 1
    assert [row["id"] for row in unseen] == ["fresh"]


def test_vicrop_merge_evaluations_keeps_all_paired_cases():
    vicrop = _load_vicrop()

    merged = vicrop.merge_evaluations(
        {"cases": [{"id": "a", "correct": True}]},
        {"cases": [{"id": "b", "correct": False}]},
    )

    assert merged["n"] == 2
    assert merged["correct"] == 1
    assert merged["accuracy"] == 0.5


def test_split_permutation_is_deterministic_and_label_blind():
    runner = _load_runner()
    rows = [{"id": str(index)} for index in range(12)]

    first = runner.shuffled_rows(rows, 7)
    second = runner.shuffled_rows(rows, 7)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert {row["id"] for row in first} == {row["id"] for row in rows}


def test_vstar_adapter_keeps_two_choice_questions():
    downloader = _load_downloader()
    question, options = downloader._vstar_question_and_options(
        "Is the mug left or right of the plate?\n(A) left\n(B) right\n"
        "Answer with the option's letter from the given choices directly."
    )

    assert question == "Is the mug left or right of the plate?"
    assert options == ["left", "right"]


def test_reservoir_rows_preserve_upstream_indices():
    downloader = _load_downloader()
    rows = [{"value": index} for index in range(40)]

    first = downloader.reservoir_rows(rows, n=8, scan=40, seed=5)
    second = downloader.reservoir_rows(rows, n=8, scan=40, seed=5)

    assert first == second
    assert all(source_index == row["value"] for source_index, row in first)


def test_vstar_method_comparison_does_not_mislabel_vicrop():
    runner = _load_runner()
    row = {"metadata": {"paper_oracle_bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2]}}

    comparison = runner.method_comparison(
        "vstar_bench", [row], selection=type("S", (), {"best": None})(), oracle=None
    )

    assert comparison is not None
    assert comparison["paper_method"].startswith("SEAL")
    assert "ViCrop" not in comparison["black_box_limitation"]


def test_visual_search_parser_accepts_tight_box_and_clamps_side():
    runner = _load_runner()

    box = runner.VLMEndpoint._parse_visual_search_box(
        '{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}',
        min_side=0.25,
        max_side=0.70,
    )

    assert box == (0.2, 0.30000000000000004, 0.25)


def test_visual_search_parser_normalizes_pixel_box():
    runner = _load_runner()

    box = runner.VLMEndpoint._parse_visual_search_box(
        '{"x1": 180, "y1": 495, "x2": 230, "y2": 557}',
        min_side=0.1,
        max_side=0.70,
        image_size=(1000, 1000),
    )

    assert box == (0.20500000000000002, 0.526, 0.1)
