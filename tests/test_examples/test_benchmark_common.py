"""examples/benchmark: the (modality x family x size) matrix, its scorers, manifest
glue, CLI no-model paths, and the generated leaf compose files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).parents[2] / "examples" / "benchmark"


@pytest.fixture(scope="module")
def common():
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import _common  # noqa: F401
    from _common import models, run, scoring, tasks

    return models, tasks, scoring, run


def test_every_matrix_cell_resolves_to_a_registered_spec_of_the_right_modality(common):
    from evalvitals.specs import get_spec

    models, *_ = common
    cells = list(models.cells())
    assert len(cells) == 19
    for modality, family, size in cells:
        resolved = models.resolve(size.key, modality)
        spec = get_spec(resolved.spec_key)
        assert resolved.family is family and resolved.size is size
        if modality == "vlm":
            assert spec.is_vlm, (size.key, spec.key)
        elif modality == "alm":
            assert spec.audio is not None, (size.key, spec.key)
        else:
            assert "text" in spec.modalities
        # the standing decision: thinking OFF on every model the benchmark runs
        assert spec.chat_template_kwargs.get("enable_thinking") is not True, spec.key


def test_the_matrix_is_the_one_specified(common):
    models, *_ = common
    by_modality = {}
    for modality, family, size in models.cells():
        by_modality.setdefault(modality, {}).setdefault(family.key, []).append(size.key)
    assert by_modality == {
        "vlm": {"qwen": ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-omni-30b-a3b"]},
        "llm": {"qwen": ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-4b"]},
        "alm": {"qwen": ["qwen3-omni-30b-a3b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-omni-30b-a3b"]},
    }


def test_nemotron_hf_local_is_bf16_and_endpoint_is_the_fp8_export(common):
    from evalvitals.specs import get_spec

    models, *_ = common
    for size_key, modality in (("nemotron-3-nano-4b", "llm"), ("nemotron-3-nano-omni-30b-a3b", "alm")):
        local = models.resolve(size_key, modality, "hf_local")
        endpoint = models.resolve(size_key, modality, "endpoint")
        assert get_spec(local.spec_key).hf_repo.endswith("-BF16")
        assert get_spec(endpoint.spec_key).hf_repo.endswith("-FP8")
        assert get_spec(endpoint.spec_key).caveats[0].startswith("ENDPOINT ONLY")
    # families without an endpoint override fall back to the hf_local spec
    assert models.resolve("qwen3.5-2b", "vlm", "endpoint").spec_key == "qwen3.5-2b-vl"


def test_nemotron_sizes_default_to_eager_attention(common):
    """The remote NemotronH code has no SDPA dispatch (transformers raises on
    attn_implementation=sdpa, 2026-08-21); every other size keeps sdpa."""
    models, *_ = common
    for size in models.SIZES.values():
        assert size.attn_impl == ("eager" if size.family == "nemotron" else "sdpa"), size.key


def test_two_gpu_sizes_default_to_device_auto(common):
    models, *_ = common
    assert models.SIZES["qwen3-omni-30b-a3b"].default_device == "auto"
    assert models.SIZES["nemotron-3-nano-omni-30b-a3b"].default_device == "auto"
    assert models.SIZES["gemma-4-12b"].default_device == "cuda"


def test_resolve_accepts_a_raw_spec_key_and_rejects_unknown_models(common):
    models, *_ = common
    raw = models.resolve("qwen2.5-vl-7b-instruct", "vlm")
    assert raw.spec_key == "qwen2.5-vl-7b-instruct" and raw.size is None and raw.family is None
    with pytest.raises(KeyError):
        models.resolve("qwen3.5-8b", "vlm")  # the 8B is the 9B; no silent alias
    with pytest.raises(ValueError):
        models.resolve("qwen3.5-2b", "alm")


def test_scorers_by_task_kind(common):
    _, _, scoring, _ = common
    score = scoring.score_output
    assert score("exact_or_numeric", "Values: 40 and 2\nFinal answer: 42", ["42"], numeric_tolerance=0.05)
    assert score("exact_or_numeric", "Final answer: 6.8%", ["6.8"], numeric_tolerance=0.05)
    assert score("exact_or_numeric", "105", ["100"], numeric_tolerance=0.05)
    assert not score("exact_or_numeric", "106", ["100"], numeric_tolerance=0.05)
    assert score("multiple_choice_letter", "I think A but the answer: (B)", ["B"], choices=["A", "B", "C", "D"])
    assert score("multiple_choice_letter", "C", ["C"], choices=["A", "B", "C", "D"])
    assert not score("multiple_choice_letter", "", ["C"], choices=["A", "B", "C", "D"])
    assert score("yes_no", "Yes, a dog barks.", ["Yes"]) and score("yes_no", "NO", ["No"])
    assert not score("yes_no", "maybe", ["No"])
    with pytest.raises(ValueError):
        score("bogus", "x", ["x"])


def test_build_cases_from_a_manifest_and_score_through_case_metadata(common, tmp_path):
    _, tasks, _, _ = common
    from PIL import Image

    (tmp_path / "images").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "a.png")
    rows = [
        {"id": "c1", "prompt": "How many?", "image": "images/a.png", "audio": None,
         "answers": ["3", "three"], "task": "exact_or_numeric", "numeric_tolerance": 0.05,
         "metadata": {"source": "t"}},
        {"id": "c2", "prompt": "How many?", "image": "images/a.png", "audio": None,
         "answers": ["10"], "task": "exact_or_numeric", "numeric_tolerance": 0.05, "metadata": {}},
    ]
    tasks.write_manifest(tmp_path / "manifest.json", rows)
    batch, loaded = tasks.build_cases(tasks.get("chartqa"), tmp_path / "manifest.json", limit=0)
    cases = list(batch)
    assert len(cases) == len(loaded) == 2
    assert cases[0].expected == ["3", "three"] and cases[0].inputs.image.endswith("a.png")
    assert cases[0].metadata["task"] == "exact_or_numeric" and cases[0].metadata["dataset"] == "chartqa"
    assert tasks.score_case(cases[0], "Answer: three") and tasks.score_case(cases[1], "10.4")
    assert not tasks.score_case(cases[1], "12")
    batch_one, _ = tasks.build_cases(tasks.get("chartqa"), tmp_path / "manifest.json", limit=1)
    assert len(list(batch_one)) == 1
    rows[0]["image"] = "images/missing.png"
    tasks.write_manifest(tmp_path / "manifest.json", rows)
    with pytest.raises(FileNotFoundError):
        tasks.build_cases(tasks.get("chartqa"), tmp_path / "manifest.json")


def test_mc_and_yes_no_cases_carry_the_output_contract(common, tmp_path):
    _, tasks, _, _ = common
    rows = [{"id": "m1", "prompt": "Which?", "image": None, "audio": None, "answers": ["B"],
             "choices": ["A", "B", "C", "D"], "task": "multiple_choice_letter", "metadata": {}}]
    tasks.write_manifest(tmp_path / "manifest.json", rows)
    (case,) = list(tasks.build_cases(tasks.get("mmau"), tmp_path / "manifest.json")[0])
    assert case.expected == "B" and case.metadata["output_contract"]["kind"] == "multiple_choice_letter"
    assert case.metadata["choices"] == ["A", "B", "C", "D"]
    assert tasks.score_case(case, "Final: B") and not tasks.score_case(case, "Final: D")


def test_llm_task_names_match_the_dataset_selection_catalog(common):
    _, tasks, _, _ = common
    from _common.tasks import llm

    mod, band = llm.catalog()
    assert set(llm.DATASETS) == {e.name for e in mod.CATALOG}
    assert {t.name for t in llm.TASKS} == set(llm.DATASETS)
    # the dataset's own grader on the extracted answer — not a verbatim-substring check
    assert llm.grade("bbh_causal_judgement", "Reasoning...\nAnswer: Yes", "Yes")
    assert not llm.grade("bbh_causal_judgement", "Answer: No", "Yes")


def test_default_tasks_and_pinned_sets(common):
    _, tasks, _, _ = common
    for modality, name in tasks.DEFAULT_TASK.items():
        assert tasks.get(name).modality == modality
    for task in tasks.TASKS.values():
        assert task.pinned_m1 and task.kind in {"exact_or_numeric", "multiple_choice_letter", "yes_no", "llm_graded"}
        assert task.download is not None and callable(task.protocol)


def test_cli_no_model_paths(common, capsys):
    _, _, _, run = common
    assert run.main(["--smoke-test"]) == 0
    assert "Smoke test passed" in capsys.readouterr().out
    assert run.main(["--list"]) == 0
    assert "nemotron-3-nano-omni-30b-a3b" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run.main(["--modality", "vlm", "--model", "qwen3.5-2b", "--dataset", "mmau", "--no-download"])


def test_leaf_compose_files_cover_every_cell_and_pin_opus5(common):
    import yaml

    models, *_ = common
    seen = set()
    for modality, family, size in models.cells():
        leaf = BENCH / modality / family.key
        compose = yaml.safe_load((leaf / "docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]
        assert size.key in services, (modality, family.key, size.key)
        svc = services[size.key]
        cmd = svc["command"]
        assert f"--model {size.key}" in cmd and f"--modality {modality}" in cmd
        assert "--judge-provider claude --judge-model claude-opus-5 --judge-effort high" in cmd
        assert ("--device auto" in cmd) == (size.gpus > 1)
        assert svc["extends"]["file"] == "../../_common/compose/base.yml"
        assert (leaf / svc["extends"]["file"]).resolve().is_file()
        assert svc["build"]["target"] == family.docker_target and svc["image"] == family.image
        assert (leaf / "README.md").is_file() and (leaf / ".env").is_symlink()
        seen.add((modality, family.key, size.key))
    assert len(seen) == 19


def test_dockerfile_has_one_stage_per_family(common):
    models, *_ = common
    text = (BENCH / "docker" / "Dockerfile").read_text(encoding="utf-8")
    for family in models.FAMILIES.values():
        assert f" AS {family.docker_target}\n" in text
    assert "FROM base AS qwen" in text and "FROM base AS gemma" in text
    # nemotron is its own stack: prebuilt Mamba kernels stop at torch 2.10
    assert "FROM python:3.11-slim AS nemotron" in text and "mamba_ssm-" in text
    assert "transformers==5.15.0" in text


def test_generation_settings_sample_only_for_llm_tasks(common):
    """Greedy Qwen3.5 (thinking off too) loops to the cap on free-form reasoning
    prompts (8/8 causal-judgement items, 2026-08-21): llm tasks sample like
    llm_benchmark; short-answer tasks stay greedy; --temperature overrides."""
    _, tasks, _, run = common
    from _common.runner import generation_settings

    parse = run.build_parser().parse_args
    llm = generation_settings(tasks.get("bbh_causal_judgement"), parse(["--modality", "llm", "--model", "qwen3.5-2b"]))
    assert llm == {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    vlm = generation_settings(tasks.get("chartqa"), parse(["--modality", "vlm", "--model", "qwen3.5-2b"]))
    assert vlm == {"max_new_tokens": 64, "do_sample": False}
    alm = generation_settings(tasks.get("mmau"), parse(["--modality", "alm", "--model", "gemma-4-e2b"]))
    assert alm == {"max_new_tokens": 64, "do_sample": False}
    forced = generation_settings(tasks.get("chartqa"), parse(
        ["--modality", "vlm", "--model", "qwen3.5-2b", "--temperature", "0.3", "--max-new-tokens", "32"]))
    assert forced["do_sample"] and forced["temperature"] == 0.3 and forced["max_new_tokens"] == 32
    greedy_llm = generation_settings(tasks.get("bbh_causal_judgement"), parse(
        ["--modality", "llm", "--model", "qwen3.5-2b", "--temperature", "0"]))
    assert greedy_llm == {"max_new_tokens": 2048, "do_sample": False}
