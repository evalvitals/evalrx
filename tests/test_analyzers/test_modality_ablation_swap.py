"""modality_ablation(mode="swap"): donors are matched on media CONTENT, not case id.

POPE asks two questions per COCO image and the manifest keeps the pair
adjacent; the old first-non-self donor rule "swapped" the same image back in
for the first pair and gave every other case one shared donor.
"""

from __future__ import annotations

import shutil

from evalrx.analyzers.perturbation.modality_ablation import ModalityAblationAnalyzer, _media_key
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model


class _ByImage(Model):
    """Answers with the bytes of the image it was given, so a real swap always changes the answer."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, inputs, **kwargs):
        self.calls += 1
        if inputs.image is None:
            return "none"
        with open(inputs.image, "rb") as fh:
            return fh.read().decode()

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _images(tmp_path, names):
    paths = {}
    for name in names:
        p = tmp_path / f"{name}.jpg"
        p.write_text(name)
        paths[name] = str(p)
    return paths


def _case(i, image, label=Label.FAIL):
    return FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}", image=image),
                       expected="Yes", label=label)


def test_swap_never_reuses_the_same_image(tmp_path):
    img = _images(tmp_path, ["a", "b", "c"])
    # POPE layout: two questions per image, adjacent
    batch = CaseBatch([_case(0, img["a"]), _case(1, img["a"]), _case(2, img["b"]),
                       _case(3, img["b"]), _case(4, img["c"]), _case(5, img["c"])])
    f = ModalityAblationAnalyzer(mode="swap").run(_ByImage(), batch).findings
    rows = {r["sample_id"]: r for r in f["per_case"]}
    image_of = {c.id: c.inputs.image for c in batch}
    for cid, row in rows.items():
        assert row["ablation_image_is_swap"] is True
        assert image_of[row["ablation_image_donor"]] != image_of[cid]
        assert row["grounded_in_image"] is True  # a different image always moves this model
    # next different-content case in batch order, wrapping: donors spread over the batch
    assert rows["c0"]["ablation_image_donor"] == "c2"
    assert rows["c1"]["ablation_image_donor"] == "c2"
    assert rows["c5"]["ablation_image_donor"] == "c0"
    assert f["ablation_mode_image"] == "swap" and f["n_drop_fallback_image"] == 0
    assert f["ungrounded_rate_image"] == 0.0


def test_identical_content_under_two_paths_is_the_same_media(tmp_path):
    img = _images(tmp_path, ["a", "b"])
    copy = tmp_path / "a_copy.jpg"
    shutil.copy(img["a"], copy)
    batch = CaseBatch([_case(0, img["a"]), _case(1, str(copy)), _case(2, img["b"])])
    rows = {r["sample_id"]: r for r in ModalityAblationAnalyzer(mode="swap").run(_ByImage(), batch).findings["per_case"]}
    assert rows["c0"]["ablation_image_donor"] == "c2"
    assert _media_key(img["a"]) == _media_key(str(copy)) != _media_key(img["b"])


def test_no_different_media_falls_back_to_drop_and_says_so(tmp_path):
    img = _images(tmp_path, ["a", "b"])
    batch = CaseBatch([_case(0, img["a"]), _case(1, img["a"])])
    f = ModalityAblationAnalyzer(mode="swap").run(_ByImage(), batch).findings
    assert f["ablation_mode_image"] == "drop" and f["n_drop_fallback_image"] == 2
    assert all(r["ablation_image_donor"] is None and r["ablation_image_is_swap"] is False
               for r in f["per_case"])

    mixed = CaseBatch([_case(0, img["a"]), _case(1, img["a"]), _case(2, img["b"])])
    f = ModalityAblationAnalyzer(mode="swap").run(_ByImage(), mixed).findings
    assert f["ablation_mode_image"] == "swap"  # c2 swaps with c0, c0/c1 with c2
    assert f["n_drop_fallback_image"] == 0


def test_drop_mode_output_is_unchanged(tmp_path):
    img = _images(tmp_path, ["a", "b"])
    batch = CaseBatch([_case(0, img["a"]), _case(1, img["b"], Label.PASS)])
    f = ModalityAblationAnalyzer().run(_ByImage(), batch).findings
    assert f["ablation_mode_image"] == "drop"
    assert "n_drop_fallback_image" not in f
    for row in f["per_case"]:
        assert set(row) == {"sample_id", "grounded_in_image", "ablation_image_is_swap"}


def test_in_memory_images_are_compared_by_pixels():
    from PIL import Image

    a = Image.new("RGB", (4, 4), (255, 0, 0))
    a2 = Image.new("RGB", (4, 4), (255, 0, 0))
    b = Image.new("RGB", (4, 4), (0, 0, 255))
    assert _media_key(a) == _media_key(a2) != _media_key(b)


# ── counterfactual donors ──────────────────────────────────────────────────────


def _q(i, image, obj, gold, label=Label.PASS):
    return FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"Is there a {obj} in the image?", image=image),
                       expected=gold, label=label, metadata={"gold": gold})


class _Says(Model):
    """Gives one fixed answer, as a model with a strong yes/no prior would."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, inputs, **kwargs):
        return self.answer

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _pope_pool(img):
    # per image one Yes and one No question, adjacent; each question repeats across images
    return CaseBatch([_q(0, img["a"], "cat", "Yes"), _q(1, img["a"], "dog", "No"),
                      _q(2, img["b"], "cat", "No"), _q(3, img["b"], "dog", "Yes"),
                      _q(4, img["c"], "cat", "Yes"), _q(5, img["c"], "dog", "No"),
                      _q(6, img["d"], "cat", "No"), _q(7, img["d"], "dog", "Yes")])


def test_counterfactual_donor_needs_a_gold_other_than_the_models_answer(tmp_path):
    img = _images(tmp_path, ["a", "b", "c", "d"])
    pool = _pope_pool(img)
    batch = CaseBatch([list(pool)[i] for i in (1, 2, 5)])  # the probed sample
    swap = ModalityAblationAnalyzer(mode="swap", donor_rule="counterfactual", donor_pool=pool)

    # answered "Yes" where the object is absent (a FAIL): a correct answer only has to
    # change on media where the same question's gold is "No" -- not on its own gold's opposite
    f = swap.run(_Says("Yes"), batch).findings
    rows = {r["sample_id"]: r for r in f["per_case"]}
    assert {k: r["ablation_image_donor"] for k, r in rows.items()} == {"c1": "c5", "c2": "c6", "c5": "c1"}
    assert {r["ablation_image_donor_expected"] for r in rows.values()} == {"no"}
    assert all(r["ablation_image_donor_kind"] == "counterfactual" for r in rows.values())
    assert all(r["grounded_in_image"] is False for r in rows.values())  # it said Yes again
    assert f["n_counterfactual_image"] == 3 and f["donor_rule"] == "counterfactual"

    # answered "No": the donor must be one where the same question's gold is "Yes"
    row = swap.run(_Says("No"), CaseBatch([list(pool)[1]])).findings["per_case"][0]
    assert row["ablation_image_donor"] == "c3" and row["ablation_image_donor_expected"] == "yes"


def test_counterfactual_falls_back_to_content_and_says_so(tmp_path):
    img = _images(tmp_path, ["a", "b"])
    pool = CaseBatch([_q(0, img["a"], "cat", "Yes"),
                      _q(1, img["b"], "cat", "Yes"),   # same question, gold = the model's answer
                      _q(2, img["a"], "cat", "No")])   # gold differs, but the SAME image
    batch = CaseBatch([list(pool)[0]])
    row = ModalityAblationAnalyzer(mode="swap", donor_rule="counterfactual",
                                   donor_pool=pool).run(_Says("Yes"), batch).findings["per_case"][0]
    assert row["ablation_image_donor_kind"] == "content" and row["ablation_image_donor"] == "c1"
    assert row["ablation_image_donor_expected"] is None


def test_content_rule_never_reports_a_counterfactual(tmp_path):
    img = _images(tmp_path, ["a", "b"])
    batch = CaseBatch([_q(0, img["a"], "cat", "Yes"), _q(1, img["b"], "cat", "No")])
    f = ModalityAblationAnalyzer(mode="swap").run(_ByImage(), batch).findings
    assert {r["ablation_image_donor_kind"] for r in f["per_case"]} == {"content"}
    assert f["n_counterfactual_image"] == 0 and f["donor_rule"] == "content"
