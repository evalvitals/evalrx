"""Tests for FailureCase, CaseBatch, and as_casebatch normalisation."""

from __future__ import annotations

import pytest

from evalrx.core import CaseBatch, FailureCase, Inputs, Label, as_casebatch
from evalrx.core.case import Provenance, Source


def test_failurecase_from_prompt():
    case = FailureCase.from_prompt("hello")
    assert case.inputs.prompt == "hello"
    assert case.label == Label.UNKNOWN
    assert len(case.id) == 12


def test_as_casebatch_from_str():
    batch = as_casebatch("a prompt")
    assert isinstance(batch, CaseBatch)
    assert len(batch) == 1
    assert batch[0].inputs.prompt == "a prompt"


def test_as_casebatch_from_failurecase():
    case = FailureCase.from_prompt("x")
    batch = as_casebatch(case)
    assert len(batch) == 1
    assert batch[0] is case


def test_as_casebatch_from_inputs():
    batch = as_casebatch(Inputs(prompt="y"))
    assert len(batch) == 1
    assert batch[0].inputs.prompt == "y"


def test_as_casebatch_from_list():
    batch = as_casebatch(["a", "b", "c"])
    assert len(batch) == 3
    assert [c.inputs.prompt for c in batch] == ["a", "b", "c"]


def test_as_casebatch_passthrough():
    original = CaseBatch.from_prompts(["a", "b"])
    assert as_casebatch(original) is original


def test_as_casebatch_rejects_unknown():
    with pytest.raises(TypeError):
        as_casebatch(object())


def test_casebatch_filter_by_label():
    batch = CaseBatch(
        [
            FailureCase.from_prompt("a", label=Label.FAIL),
            FailureCase.from_prompt("b", label=Label.PASS),
            FailureCase.from_prompt("c", label=Label.FAIL),
        ]
    )
    fails = batch.filter(label=Label.FAIL)
    assert len(fails) == 2


def test_stratified_head_keeps_minority_fails():
    """A capped subsample must keep FAIL representation, not just the head
    (which on an enriched batch is mostly PASS). Defect 8."""
    # 4 FAIL up front, then 20 PASS — a plain [:8] head would grab 4 fail + 4
    # pass, but an enriched batch interleaves; build the adversarial layout:
    cases = ([FailureCase.from_prompt(f"p{i}", label=Label.PASS) for i in range(20)]
             + [FailureCase.from_prompt(f"f{i}", label=Label.FAIL) for i in range(8)])
    batch = CaseBatch(cases)
    kept = batch.stratified_head(8)
    n_fail = sum(1 for c in kept if c.label == Label.FAIL)
    assert len(kept) == 8
    assert n_fail == 4          # half the budget goes to the minority class
    # document order preserved among kept cases
    assert [c.inputs.prompt for c in kept] == sorted(
        [c.inputs.prompt for c in kept], key=lambda p: (p[0] != "p"))


def test_stratified_head_noop_when_budget_exceeds_size():
    batch = CaseBatch([FailureCase.from_prompt("a", label=Label.FAIL),
                       FailureCase.from_prompt("b", label=Label.PASS)])
    assert len(batch.stratified_head(10)) == 2
    assert len(batch.stratified_head(0)) == 2


def test_casebatch_filter_by_tags():
    batch = CaseBatch(
        [
            FailureCase.from_prompt("a", tags={"hallucination"}),
            FailureCase.from_prompt("b", tags={"format"}),
        ]
    )
    assert len(batch.filter(tags={"hallucination"})) == 1


def test_failurecase_to_dict_roundtrip_shape():
    case = FailureCase.from_prompt(
        "q",
        expected="gold",
        label=Label.FAIL,
        tags={"x"},
        provenance=Provenance(source=Source.AGENT),
    )
    d = case.to_dict()
    assert d["inputs"]["prompt"] == "q"
    assert d["label"] == "fail"
    assert d["tags"] == ["x"]
    assert d["provenance"]["source"] == "agent"


def test_inputs_carry_omni_modalities():
    # omni inputs: image (VLM) + audio + video slots alongside the prompt.
    inp = Inputs(prompt="describe", image="<img>", audio="<wav>", video="<frames>")
    case = FailureCase(inputs=inp)
    d = case.to_dict()
    assert d["inputs"]["audio"] == "<wav>" and d["inputs"]["video"] == "<frames>"
    assert d["inputs"]["image"] == "<img>"
    # defaults stay None for a text-only case
    assert Inputs(prompt="x").audio is None and Inputs(prompt="x").video is None
