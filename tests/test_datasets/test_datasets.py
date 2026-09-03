"""Dataset loaders → CaseBatch + answer verifiers."""

from __future__ import annotations

import json

import pytest

from evalrx.core.case import CaseBatch
from evalrx.datasets import (
    GUIOSDataset,
    LLMQADataset,
    PureQADataset,
    Spatial457Dataset,
    TextVQASizeDataset,
    VLMQADataset,
    WebSearchQADataset,
    contains_answer,
    exact_match,
    normalize,
)


def test_llm_qa_sample_loads():
    cb = LLMQADataset.sample().load()
    assert isinstance(cb, CaseBatch) and len(cb) == 4
    c = cb[0]
    assert c.inputs.prompt.startswith("What is the capital") and c.expected == "Paris"
    assert "llm_qa" in c.tags and c.metadata["difficulty"] == "easy"


def test_pure_qa_is_llm_qa_alias():
    assert PureQADataset is LLMQADataset  # back-compat


def test_pure_qa_from_records_and_jsonl(tmp_path):
    recs = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    cb = PureQADataset.from_records(recs).load()
    assert len(cb) == 2 and cb[1].expected == "a2"

    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    cb2 = PureQADataset.from_jsonl(str(p)).load()
    assert len(cb2) == 2 and cb2[0].inputs.prompt == "q1"


def test_web_search_and_gui_os_tags():
    w = WebSearchQADataset.sample().load()
    assert "web_search" in w[0].tags and w[0].metadata["requires_retrieval"] is True
    g = GUIOSDataset.sample().load()
    assert "gui_os" in g[0].tags and g[0].expected == {"dark_mode": True}
    assert g[0].inputs.prompt.startswith("Open Settings")


def test_verifiers():
    assert exact_match("Paris", " paris ")
    assert contains_answer("The capital is Paris.", "Paris")
    assert not contains_answer("It is Lyon.", "Paris")
    assert normalize("  Hello   World ") == "hello world"


def test_vlm_qa_image_goes_to_inputs_not_metadata():
    recs = [{"question": "what color?", "answer": "red", "image": "<PIL>", "scene": "s1"}]
    cb = VLMQADataset.from_records(recs, meta_keys=("scene",)).load()
    c = cb[0]
    assert "vlm_qa" in c.tags
    assert c.inputs.image == "<PIL>" and c.expected == "red"
    assert c.metadata == {"scene": "s1"}          # image NOT dumped into metadata
    assert "image" not in c.metadata


def test_spatial457_subset_validation():
    import pytest
    with pytest.raises(ValueError):
        Spatial457Dataset(subset="not_a_subset")
    assert "L5_6d_spatial" in Spatial457Dataset.SUBTYPES and len(Spatial457Dataset.SUBTYPES) == 7


def test_spatial457_record_mapping():
    cb = Spatial457Dataset.from_records([
        {"image": "<img>", "image_filename": "superCLEVR_new_000001.png",
         "question_index": 100001, "question": "Is the red object in front of the car?", "answer": "True"},
    ], subset="L5_6d_spatial").load()
    c = cb[0]
    assert c.inputs.prompt.startswith("Is the red object") and c.expected == "True"
    assert c.inputs.image == "<img>"
    assert {"vlm_qa", "spatial457", "L5_6d_spatial"}.issubset(c.tags)
    assert c.metadata["image_filename"] == "superCLEVR_new_000001.png"
    assert c.metadata["question_index"] == 100001 and c.metadata["subset"] == "L5_6d_spatial"


def test_spatial457_sample():
    cb = Spatial457Dataset.sample().load()
    assert len(cb) == 2 and "spatial457" in cb[0].tags
    assert cb[0].metadata["dataset"] == "spatial457"


# ---------------- TextVQA size split ----------------

def test_textvqa_size_dataset_partitions_by_answer_bbox_area():
    recs = [
        {
            "question_id": "q_small",
            "image": "small.jpg",
            "image_width": 1000,
            "image_height": 1000,
            "question": "What word is on the tiny sign?",
            "answers": ["moma", "MoMA"],
            "answer_bbox": [100, 120, 40, 12],  # 0.00048
        },
        {
            "question_id": "q_medium",
            "image": "medium.jpg",
            "image_width": 1000,
            "image_height": 1000,
            "question": "What is printed on the poster?",
            "answers": ["sale"],
            "answer_bbox": [100, 100, 100, 100],  # 0.01
        },
        {
            "question_id": "q_large",
            "image": "large.jpg",
            "image_width": 1000,
            "image_height": 1000,
            "question": "What brand is shown?",
            "answers": ["acme"],
            "answer_bbox": [100, 100, 300, 300],  # 0.09
        },
    ]

    small = TextVQASizeDataset.from_records(recs, size_split="small").load()
    medium = TextVQASizeDataset.from_records(recs, size_split="medium").load()
    large = TextVQASizeDataset.from_records(recs, size_split="large").load()

    assert [c.id for c in small] == ["q_small"]
    assert [c.id for c in medium] == ["q_medium"]
    assert [c.id for c in large] == ["q_large"]
    c = small[0]
    assert c.expected == {"any_of": ["moma", "MoMA"]}
    assert c.metadata["dataset"] == "textvqa"
    assert c.metadata["size_split"] == "small"
    assert c.metadata["answer_bbox_xyxy_norm"] == [0.1, 0.12, 0.14, 0.132]
    assert {"vlm_qa", "textvqa", "textvqa_small"}.issubset(c.tags)


def test_textvqa_size_dataset_reads_image_dimensions_from_path(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(img_path)
    recs = [
        {
            "question_id": "q_img",
            "image": "img.jpg",
            "question": "What is the small label?",
            "answer": "abc",
            "answer_bbox": [20, 10, 10, 5],
        }
    ]

    cb = TextVQASizeDataset.from_records(
        recs,
        image_dir=tmp_path,
        size_split="small",
    ).load()

    assert len(cb) == 1
    assert cb[0].inputs.image == str(img_path)
    assert cb[0].metadata["bbox_area_ratio"] == pytest.approx(0.0025)


def test_textvqa_size_dataset_accepts_dict_answer_and_normalized_xyxy():
    recs = [
        {
            "question_id": "q_norm",
            "image_width": 1000,
            "image_height": 1000,
            "question": "What word is shown?",
            "answer": {"text": "exit"},
            "answer_bbox": [0.1, 0.2, 0.14, 0.23],
        }
    ]

    cb = TextVQASizeDataset.from_records(recs, size_split="small").load()

    assert len(cb) == 1
    assert cb[0].expected == "exit"
    assert cb[0].metadata["answer_bbox_xyxy_norm"] == [0.1, 0.2, 0.14, 0.23]


def test_textvqa_size_dataset_keeps_degenerate_small_bbox():
    recs = [
        {
            "question_id": "q_zero",
            "image_width": 1000,
            "image_height": 500,
            "question": "What word is shown?",
            "answer": "unknown",
            "answer_bbox": [100, 200, 0, 0],
            "bbox_format": "xywh",
        }
    ]

    cb = TextVQASizeDataset.from_records(recs, size_split="small").load()

    assert len(cb) == 1
    assert cb[0].metadata["bbox_degenerate"] is True
    assert cb[0].metadata["answer_bbox_xyxy_norm"] == [0.1, 0.4, 0.101, 0.402]


# ---------------- VQA-RAD ----------------

def _rad_records():
    """Synthetic VQA-RAD-shaped records: 3 easy, 3 gold-yes, 3 gold-no, 1 open."""
    recs = [
        {"image": "<img>", "question": "what imaging modality was used?", "answer": "ct"},
        {"image": "<img>", "question": "which plane is this image taken in?", "answer": "axial"},
        {"image": "<img>", "question": "what organ system is shown?", "answer": "chest"},
        {"image": "<img>", "question": "what is the largest structure?", "answer": "liver"},  # other
    ]
    for i in range(3):
        recs.append({"image": "<img>", "question": f"is finding {i} present?", "answer": "yes"})
        recs.append({"image": "<img>", "question": f"is lesion {i} visible?", "answer": "no"})
    return recs


def test_vqa_rad_categorize():
    from evalrx.datasets.vlm_qa import _categorize_vqa_rad
    assert _categorize_vqa_rad("what imaging modality was used?", "ct") == "easy"
    assert _categorize_vqa_rad("which plane is this?", "axial") == "easy"
    assert _categorize_vqa_rad("is there a pneumothorax?", "no") == "presence"
    assert _categorize_vqa_rad("is the heart enlarged?", "yes") == "presence"
    assert _categorize_vqa_rad("what is the largest structure?", "liver") == "other"


def test_vqa_rad_balanced_mix_and_pope_labels():
    from evalrx.datasets import VQARADDataset

    cb = VQARADDataset.from_records(_rad_records(), n_easy=2, n_presence=4, seed=0).load()
    easy = [c for c in cb if c.metadata["category"] == "easy"]
    pres = [c for c in cb if c.metadata["category"] == "presence"]
    assert len(easy) == 2 and len(pres) == 4

    # Presence cases: balanced gold, pope_label set, dict expected rubric.
    golds = [c.metadata["pope_label"] for c in pres]
    assert golds.count("yes") == 2 and golds.count("no") == 2
    for c in pres:
        assert c.inputs.prompt.endswith("Answer yes or no.")
        gold = c.metadata["pope_label"]
        other = "no" if gold == "yes" else "yes"
        assert c.expected == {"all_of": [gold], "none_of": [other]}
        assert "med_vqa" in c.tags and "presence" in c.tags

    # Easy cases: token any_of rubric, no pope_label.
    for c in easy:
        assert isinstance(c.expected, dict) and c.expected.get("any_of")
        assert "pope_label" not in c.metadata
        assert c.inputs.prompt.endswith("Answer briefly.")


def test_vqa_rad_easy_rubric_tokenizes_messy_gold():
    from evalrx.datasets.vlm_qa import _easy_answer_rubric
    assert _easy_answer_rubric("xray - plain film") == {"any_of": ["xray", "plain", "film"]}
    # tab-separated organ lists + plural tolerance
    r = _easy_answer_rubric("respiratory \tcardia c\tmusculoskeletal")
    assert "respiratory" in r["any_of"]
    assert _easy_answer_rubric("kidneys")["any_of"] == ["kidneys", "kidney"]
    assert _easy_answer_rubric("ct")["any_of"] == ["ct"]


def test_vqa_rad_deterministic_sampling():
    from evalrx.datasets import VQARADDataset

    a = VQARADDataset.from_records(_rad_records(), n_easy=2, n_presence=4, seed=0).load()
    b = VQARADDataset.from_records(_rad_records(), n_easy=2, n_presence=4, seed=0).load()
    assert [c.inputs.prompt for c in a] == [c.inputs.prompt for c in b]


def test_vqa_rad_sample_offline():
    from evalrx.datasets import VQARADDataset
    cb = VQARADDataset.sample().load()
    assert len(cb) >= 2
    assert all(c.metadata["dataset"] == "vqa_rad" for c in cb)
