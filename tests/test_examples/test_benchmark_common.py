"""examples/benchmark: the (modality x family x size) matrix, its scorers, manifest
glue, CLI no-model paths, and the generated leaf compose files."""

from __future__ import annotations

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
    assert len(cells) == 43   # 19 open-weight cells + 8 Gemini models x 3 modalities
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


def test_pinned_m1_priority_covers_audio_model_kinds(common):
    from _common.runner import _pinned_priority_override

    pinned = ["answer_extraction_audit", "termination_audit"]
    override = _pinned_priority_override(pinned)
    assert override["alm"] == pinned
    assert override["avlm"] == pinned
    assert set(override) == {"vlm", "avlm", "alm", "agent", "llm"}


def test_audiocaps_protocol_accepts_balanced_grounding_brittleness(common):
    """AudioCaps may fail symmetrically on present/absent sounds but still be
    repairable when the same audio decision changes under a semantic rephrase."""
    _, tasks, *_ = common
    pattern = tasks.TASKS["audiocaps_hallu"].protocol("test-model").failure_patterns
    assert "balanced across present and absent sounds" in pattern
    assert "meaning-preserving restatement" in pattern
    assert "trading present-sound" in pattern


def test_the_matrix_is_the_one_specified(common):
    models, *_ = common
    by_modality = {}
    for modality, family, size in models.cells():
        by_modality.setdefault(modality, {}).setdefault(family.key, []).append(size.key)
    gemini = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
              "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
    assert by_modality == {
        "vlm": {"qwen": ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-omni-30b-a3b"],
                "gemini": gemini},
        "llm": {"qwen": ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-4b"],
                "gemini": gemini},
        "alm": {"qwen": ["qwen3-omni-30b-a3b"],
                "gemma": ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b"],
                "nemotron": ["nemotron-3-nano-omni-30b-a3b"],
                "gemini": gemini},
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


def test_endpoint_backend_sends_thinking_off_and_top_k_in_extra_body(common, monkeypatch):
    """vLLM applies the repo chat template server-side with ITS defaults —
    Nemotron 3 defaults enable_thinking=True — so the endpoint runtime must
    carry the spec's chat_template_kwargs (and the top_k the OpenAI schema
    lacks) in extra_body."""
    _, _, _, run = common
    captured = {}
    import evalvitals.models.backends.openai_compat as oc

    def fake_runtime(**kw):
        captured.update(kw)
        from evalvitals.models.backends.base import RuntimeConfig
        return RuntimeConfig(generate_fn=lambda prompt, model="", **k: "ok")

    monkeypatch.setattr(oc, "openai_runtime", fake_runtime)
    from _common import models as M
    from _common import runner
    from _common import tasks as T
    args = run.build_parser().parse_args([
        "--modality", "llm", "--model", "nemotron-3-nano-4b", "--backend", "endpoint"])
    resolved = M.resolve("nemotron-3-nano-4b", "llm", backend="endpoint")
    runner.load_model(resolved, args, T.get("bbh_causal_judgement"))
    assert captured["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["extra_body"]["top_k"] == 20 and captured["top_p"] == 0.95
    assert captured["temperature"] == 0.6 and captured["max_tokens"] == 2048


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


def test_parsed_choice_refuses_leaked_thought_preambles_and_enumerations(common):
    """gemma-4-e2b/MMAU 2026-08-22: 100/256 outputs were a spontaneous ``thought``
    channel cut at the 64-token cap while restating the options; the old
    "last bare letter anywhere" fallback turned 16 of them into PASSes."""
    _, _, scoring, _ = common
    pc = scoring.parsed_choice
    preamble = ("thought\n1.  **Analyze the Request:** The user wants me to identify the source "
                "of the music. The options are (A) Radio, (B) Live")
    assert pc(preamble, "ABCD") == ""                                  # no commitment -> no letter
    assert pc("<|channel>thought\nThe options are (A) W", "ABCD") == ""
    assert pc("thought\nI listened carefully.\nAnswer: C", "ABCD") == "C"   # a tag still commits
    assert pc("(B)", "ABCD") == "B" and pc("B.", "ABCD") == "B"
    assert pc("B\n\nBecause the clip is a dog barking.", "ABCD") == "B"  # answer first, prose after
    assert pc("It is (A) dog, (B) cat or (C) cow.", "ABCD") == ""        # an enumeration is not an answer
    assert pc("Either B or C.", "ABCD") == ""                            # a hedge is not an answer
    assert pc("I think A but the answer: (B)", "ABCD") == "B"
    assert pc("A dog barks.", "ABCD") == ""                              # the article is not option A
    assert pc("I'd go with B", "ABCD") == "B" and pc("b", "ABCD") == "B"
    assert not scoring.score_output("multiple_choice_letter", preamble, ["A"], choices=["A", "B", "C", "D"])
    yn = scoring.parsed_yes_no
    assert yn("thought\nThe user asks whether a dog barks, yes or no.") == ""
    assert yn("thought\nListening...\nAnswer: No") == "No"
    assert yn("Yes, a dog barks.") == "Yes"


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
        assert task.pinned_m1 and task.kind in {
            "exact_or_numeric", "multiple_choice_letter", "yes_no", "llm_graded", "short_answer_em"}
        assert task.download is not None and callable(task.protocol)


def test_cli_no_model_paths(common, capsys):
    _, _, _, run = common
    defaults = run.build_parser().parse_args([])
    assert (defaults.judge_provider, defaults.judge_model, defaults.judge_effort) == (
        "codex", "gpt-5.6-terra", "medium"
    )
    assert defaults.fix_repair_rounds == 2
    assert defaults.allow_adapted_paper_methods is False
    assert run.build_parser().parse_args([
        "--allow-adapted-paper-methods"
    ]).allow_adapted_paper_methods is True
    assert run.build_parser().parse_args(["--fix-repair-rounds", "4"]).fix_repair_rounds == 4
    assert run.main(["--smoke-test"]) == 0
    assert "Smoke test passed" in capsys.readouterr().out
    assert run.main(["--list"]) == 0
    assert "nemotron-3-nano-omni-30b-a3b" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run.main(["--modality", "vlm", "--model", "qwen3.5-2b", "--dataset", "mmau", "--no-download"])


def test_leaf_compose_files_cover_every_cell_and_pin_codex_terra(common):
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
        assert (
            "--judge-provider codex --judge-model gpt-5.6-terra "
            "--judge-effort medium"
        ) in cmd
        assert ("--device auto" in cmd) == (size.gpus > 1)
        assert svc["extends"]["file"] == "../../_common/compose/base.yml"
        assert (leaf / svc["extends"]["file"]).resolve().is_file()
        assert svc["build"]["target"] == family.docker_target and svc["image"] == family.image
        assert (leaf / "README.md").is_file() and (leaf / ".env").is_symlink()
        seen.add((modality, family.key, size.key))
    assert len(seen) == 43


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


def test_benchmark_autofix_escalates_by_default(common):
    _, _, _, run = common
    parse = run.build_parser().parse_args
    default = parse(["--modality", "vlm", "--model", "qwen3.5-2b"])
    disabled = parse([
        "--modality", "vlm", "--model", "qwen3.5-2b", "--no-auto-escalate",
    ])

    assert default.auto_escalate is True
    assert disabled.auto_escalate is False


def test_skip_m4_is_independent_of_fix_and_tier_search(common):
    _, _, _, run = common
    args = run.build_parser().parse_args([
        "--modality", "vlm", "--model", "qwen3.5-2b", "--skip-m4",
    ])

    assert args.skip_m4 is True
    assert args.skip_fix is False
    assert args.auto_escalate is True
    assert args.fix_candidate == ""


def test_run_fix_isolated_hides_the_run_dir_from_the_fix_stage_and_restores_it(common, tmp_path):
    """The fix stage must not be able to read baseline.json / discovery_cases /
    the event log from its workspace; afterwards everything is back and the
    events the fix stage logged are appended after the pre-fix ones."""
    import importlib
    import logging

    runner = importlib.import_module("_common.runner")

    run_dir = tmp_path / "run"
    (run_dir / "logs" / "report").mkdir(parents=True)
    (run_dir / "baseline.json").write_text('[{"id": "c-0", "expected": "Yes", "label": "fail"}]')
    (run_dir / "logs" / "report" / "discovery_cases.json").write_text('[{"expected": "Yes"}]')
    log_path = run_dir / "logs" / "run_log.jsonl"
    log_path.write_text('{"event": "case_record", "expected": "Yes"}\n')

    lg = logging.getLogger("bench-quarantine-wiring-test")
    lg.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.log_path = log_path

    class Loop:
        def run_fix(self, report, cases, **kw):
            # what the coder/sandbox could reach from logs/fixes/<trial>/workspace
            assert not (run_dir / "baseline.json").exists()
            assert not (run_dir / "logs" / "report" / "discovery_cases.json").exists()
            assert log_path.read_text() == ""
            assert "expected" not in " ".join(
                p.read_text(errors="replace") for p in run_dir.rglob("*") if p.is_file())
            lg.info('{"event": "fix", "candidate": "x"}')
            handler.flush()
            (run_dir / "logs" / "fixes").mkdir()
            (run_dir / "logs" / "fixes" / "outcome.md").write_text("NOT FIXED")
            return {"kw": kw, "report": report, "cases": cases}

    try:
        outcome = runner.run_fix_isolated(Loop(), run_dir, ctx, "REPORT", ["c-0"],
                                          max_tier="L3b", auto_escalate=True)
    finally:
        lg.removeHandler(handler)
        handler.close()

    assert outcome == {"kw": {"max_tier": "L3b", "auto_escalate": True},
                       "report": "REPORT", "cases": ["c-0"]}
    assert (run_dir / "baseline.json").read_text() == '[{"id": "c-0", "expected": "Yes", "label": "fail"}]'
    assert (run_dir / "logs" / "report" / "discovery_cases.json").read_text() == '[{"expected": "Yes"}]'
    assert log_path.read_text() == ('{"event": "case_record", "expected": "Yes"}\n'
                                    '{"event": "fix", "candidate": "x"}\n')
    assert (run_dir / "logs" / "fixes" / "outcome.md").read_text() == "NOT FIXED"
    assert (run_dir / "fix_quarantine.json").exists()


def test_llm_download_freezes_from_the_hub_when_the_datasets_server_is_down(common, tmp_path, monkeypatch, capsys):
    """/filter answering 500 for hours must not leave a sliced dataset unfreezable."""
    import json

    from _common.tasks import llm

    _, band = llm.catalog()

    def _down(spec, want, seed=0, **kw):
        raise RuntimeError(f"datasets-server unavailable for {spec.dataset}")

    rows = [{"question": f"Q{i}?", "options": ["yes", "no", "maybe", "never"],
             "answer_letter": "ABCD"[i % 4], "discipline": "Law"} for i in range(30)]
    hub_calls = []

    def _hub(spec, want, seed=0, **kw):
        hub_calls.append((spec.name, want, seed))
        return list(rows)

    monkeypatch.setattr(band, "fetch_rows", _down)
    monkeypatch.setattr(band, "fetch_rows_hub", _hub)

    summary = llm.download(tmp_path, limit=20, seed=7, dataset="supergpqa_law")

    assert summary["fetch"] == "hub" and summary["kept"] == 20
    assert hub_calls == [("supergpqa_law", 20, 7)]
    text = (tmp_path / "manifest.json").read_text()
    assert text.count('"supergpqa_law-') == 20
    assert json.loads(text)  # a manifest the run can load
    out = capsys.readouterr().out
    assert "datasets-server unavailable for m-a-p/SuperGPQA" in out and "hub files" in out


# ----------------------------------------------------------------------
# Gemini: the closed-weight API family
# ----------------------------------------------------------------------
def test_gemini_sizes_run_only_on_the_gemini_backend(common):
    models, *_ = common
    from evalvitals.specs import get_spec

    resolved = models.resolve("gemini-3.6-flash", "alm")          # the default --backend is overridden
    assert resolved.backend == "gemini" and resolved.spec_key == "gemini-3.6-flash"
    assert resolved.family.key == "gemini" and resolved.size.gpus == 0
    assert models.resolve("gemini-3.6-flash", "llm", "endpoint").backend == "gemini"
    assert get_spec(resolved.spec_key).api_only
    with pytest.raises(ValueError, match="gemini family"):
        models.resolve("gemma-4-e2b", "llm", "gemini")
    # a raw api-only spec key of another provider still has exactly one way to run
    assert models.resolve("step-1o-vision", "vlm").backend == "endpoint"
    assert "backend=gemini (api)" in models.matrix_text()


def test_api_backends_clamp_the_fix_ladder_to_l2(common):
    """Every L3a repair needs a white-box executor and L3b a forward hook: on the
    API backends the ceiling is L2, whatever --fix-tier asked for."""
    from _common.runner import API_FIX_CEILING, effective_fix_tier

    assert API_FIX_CEILING == "L2"
    assert effective_fix_tier("hf_local", "L3b") == "L3b"
    assert effective_fix_tier("gemini", "L3b") == "L2"
    assert effective_fix_tier("gemini", "L3a") == "L2"
    assert effective_fix_tier("endpoint", "L3b") == "L2"
    assert effective_fix_tier("gemini", "L2") == "L2"
    assert effective_fix_tier("gemini", "L1") == "L1"


def test_cli_gemini_flags_parse_and_default_to_the_floor(common):
    _, _, _, run = common
    args = run.build_parser().parse_args([
        "--backend", "gemini", "--thinking-level", "low", "--thinking-budget", "0", "--request-retries", "8",
    ])
    assert (args.backend, args.thinking_level, args.thinking_budget, args.request_retries) == ("gemini", "low", 0, 8)
    defaults = run.build_parser().parse_args([])
    assert (defaults.thinking_level, defaults.thinking_budget, defaults.api_key) == (None, None, None)
    assert defaults.request_timeout == 300.0 and defaults.request_retries == 5
    with pytest.raises(SystemExit):
        run.build_parser().parse_args(["--thinking-level", "max"])


def test_load_model_gemini_branch_sends_the_floor_and_claims_generate_only(common, monkeypatch):
    from evalvitals.core.capability import Capability
    from evalvitals.models.backends.base import RuntimeConfig

    models, tasks, _, run = common
    from _common import runner

    captured: dict = {}

    def fake_runtime(**kw):
        captured.clear()
        captured.update(kw)
        return RuntimeConfig(generate_fn=lambda prompt, model="", **k: "Answer: Yes")

    monkeypatch.setattr("evalvitals.models.backends.gemini_compat.gemini_runtime", fake_runtime)

    # llm task: sampled decoding at the task cap, thinking at the floor, no logprobs
    args = run.build_parser().parse_args(["--modality", "llm", "--model", "gemini-3.6-flash"])
    resolved = models.resolve(args.model, args.modality, args.backend)
    model, gen_kwargs, spec = runner.load_model(resolved, args, tasks.get("bbh_causal_judgement"))
    assert gen_kwargs == {} and spec.key == "gemini-3.6-flash"
    assert model.capabilities == frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
    assert {"image", "audio"} <= set(model.modalities)
    assert captured["with_logprobs"] is False and captured["api_key"] is None
    assert captured["thinking"].floor is True and captured["thinking"].level is None
    assert (captured["temperature"], captured["top_p"], captured["top_k"]) == (0.6, 0.95, 20)
    assert captured["max_output_tokens"] == 2048
    assert (captured["timeout"], captured["retries"]) == (300.0, 5)
    assert runner.model_label(resolved).endswith("= api:gemini-3.6-flash")

    # alm task: greedy at 64 tokens; explicit level + --enable-thinking reach the policy
    args = run.build_parser().parse_args(["--modality", "alm", "--model", "gemini-2.5-flash",
                                          "--thinking-level", "low", "--enable-thinking", "--api-key", "k"])
    resolved = models.resolve(args.model, args.modality, args.backend)
    runner.load_model(resolved, args, tasks.get("mmau"))
    assert captured["temperature"] == 0.0 and "top_p" not in captured
    assert captured["max_output_tokens"] == 64 and captured["api_key"] == "k"
    assert captured["thinking"].level == "low" and captured["thinking"].floor is False



def test_pope_tasks_are_registered_per_split(common):
    _, tasks, _, _ = common
    assert [n for n in tasks.names("vlm") if n.startswith("pope_")] == [
        "pope_random", "pope_popular", "pope_adversarial"]
    for name in ("pope_random", "pope_popular", "pope_adversarial"):
        task = tasks.get(name)
        assert (task.modality, task.kind) == ("vlm", "yes_no")
        assert (task.default_limit, task.default_seed, task.max_new_tokens) == (1000, 2305, 16)
    assert tasks.DEFAULT_TASK["vlm"] == "chartqa"  # pope is opt-in, not the default


def _fake_pope_lines(split, n_images):
    """Per image: no, yes, yes, no in file order — first yes is q+2, first no is q+1."""
    lines, qid = [], 0
    for i in range(n_images):
        image = f"COCO_val2014_{i:012d}.jpg"
        for obj, label in ((f"absent-{split}-a", "no"), ("cat", "yes"),
                           ("bench", "yes"), (f"absent-{split}-b", "no")):
            qid += 1
            lines.append({"question_id": qid, "image": image,
                          "text": f"Is there a {obj} in the image?", "label": label})
    return lines


def test_pope_download_keeps_first_yes_first_no_per_image_and_reuses_sibling_images(
        common, tmp_path, monkeypatch):
    _, tasks, _, _ = common
    from _common.tasks import pope

    fetched = []
    monkeypatch.setattr(pope, "_probe_lines", lambda split, cache: _fake_pope_lines(split, 3))
    monkeypatch.setattr(pope, "_fetch_image",
                        lambda file_name, dest: (fetched.append(file_name), dest.write_bytes(b"img")))
    summary = tasks.get("pope_random").download(tmp_path / "pope_random", limit=0, seed=2305)
    assert (summary["kept"], summary["images"], summary["downloaded"]) == (6, 3, 3)
    rows = tasks.load_rows(tmp_path / "pope_random" / "manifest.json")
    # first yes (qid i*4+2) then first no (qid i*4+1) of every image, in file order
    assert [r["source_index"] for r in rows] == [2, 1, 6, 5, 10, 9]
    assert [r["answers"] for r in rows[:2]] == [["Yes"], ["No"]]
    assert rows[0]["prompt"] == "Is there a cat in the image? Answer with only the single word Yes or No."
    assert rows[1]["metadata"]["object"] == "absent-random-a"
    assert all(r["task"] == "yes_no" and r["metadata"]["split"] == "random" for r in rows)

    # the popular split shares the images: hardlinked from the sibling dir, no new fetches
    summary2 = tasks.get("pope_popular").download(tmp_path / "pope_popular", limit=0, seed=2305)
    assert (summary2["reused"], summary2["downloaded"]) == (3, 0) and fetched == fetched[:3]
    assert (tmp_path / "pope_popular" / "images" / "COCO_val2014_000000000000.jpg").is_file()

    # a smaller limit keeps whole pairs and is seed-deterministic
    a = tasks.get("pope_adversarial").download(tmp_path / "pope_adversarial", limit=4, seed=7)
    b = tasks.get("pope_adversarial").download(tmp_path / "pope_adv_again", limit=4, seed=7)
    ra = tasks.load_rows(tmp_path / "pope_adversarial" / "manifest.json")
    rb = tasks.load_rows(tmp_path / "pope_adv_again" / "manifest.json")
    assert a["kept"] == b["kept"] == 4 and [r["id"] for r in ra] == [r["id"] for r in rb]


def test_pope_cases_score_through_the_yes_no_grader(common, tmp_path, monkeypatch):
    _, tasks, _, _ = common
    from _common.tasks import pope

    monkeypatch.setattr(pope, "_probe_lines", lambda split, cache: _fake_pope_lines(split, 1))
    monkeypatch.setattr(pope, "_fetch_image", lambda file_name, dest: dest.write_bytes(b"img"))
    tasks.get("pope_adversarial").download(tmp_path / "pope_adversarial", limit=0, seed=2305)
    batch, _rows = tasks.build_cases(
        tasks.get("pope_adversarial"), tmp_path / "pope_adversarial" / "manifest.json")
    yes_case, no_case = list(batch)
    assert yes_case.metadata["gold"] == "Yes" and no_case.metadata["gold"] == "No"
    assert yes_case.inputs.image and yes_case.inputs.image.endswith(".jpg")
    assert tasks.score_case(yes_case, "Yes, there is a cat.")
    assert not tasks.score_case(yes_case, "No.")
    assert tasks.score_case(no_case, "No, I see none.")
    assert not tasks.score_case(no_case, "Yes")


def test_hotpotqa_gepa_is_registered(common):
    _, tasks, _, _ = common
    task = tasks.get("hotpotqa_gepa")
    assert task.modality == "llm" and task.kind == "short_answer_em"
    assert (task.default_limit, task.default_seed, task.max_new_tokens) == (300, 1, 512)
    assert not task.short_answer  # reason-then-'Answer:' contract, like the llm tasks
    assert "hotpotqa_gepa" in tasks.names("llm")
    assert tasks.DEFAULT_TASK["llm"] == "bbh_causal_judgement"  # opt-in, not the default


def _fake_hotpot_rows(n):
    return [
        {
            "id": f"hp{i}", "question": f"Question {i}?", "answer": "The Beatles" if i % 4 else "yes",
            "type": "bridge" if i % 2 else "comparison", "level": "hard",
            "supporting_facts": {"title": [f"T{i}B", f"T{i}A"], "sent_id": [0, 0]},
            "context": {"title": [f"T{i}A", f"T{i}B"],
                        "sentences": [[f"S{i}a. ", f"S{i}b."], [f"S{i}c."]]},
        }
        for i in range(n)
    ]


def test_hotpotqa_download_reconstructs_the_gepa_sample(common, tmp_path, monkeypatch):
    import random

    _, tasks, _, _ = common
    from _common.tasks import hotpotqa

    # 1000 fake rows: the test pool is the first 400, larger than 300 -> the
    # seed-1 sample engages, exactly GEPA's trim_dataset on the pool.
    monkeypatch.setattr(hotpotqa, "_load_train_rows", lambda: _fake_hotpot_rows(1000))
    summary = tasks.get("hotpotqa_gepa").download(tmp_path / "h", limit=0, seed=1)
    assert summary["kept"] == 300 and summary["pool"] == [0, 400]
    rows = tasks.load_rows(tmp_path / "h" / "manifest.json")
    expected = random.Random(1).sample(range(400), 300)
    assert [r["source_index"] for r in rows] == expected          # sample ORDER, not sorted
    assert [r["sample_rank"] for r in rows] == list(range(300))
    first = rows[0]
    i = expected[0]
    assert first["prompt"] == (
        f"Context:\nT{i}A: S{i}a. S{i}b.\n\nT{i}B: S{i}c.\n\nQuestion: Question {i}?\n\n"
        + hotpotqa.INSTRUCTION
    )
    assert first["answers"] == ["The Beatles" if i % 4 else "yes"]
    assert first["task"] == "short_answer_em"
    assert first["metadata"]["supporting_titles"] == [f"T{i}A", f"T{i}B"]  # sorted
    assert first["metadata"]["gepa_split"] == "test" and first["metadata"]["hotpot_id"] == f"hp{i}"

    # limit keeps the first rows of the same order
    tasks.get("hotpotqa_gepa").download(tmp_path / "h50", limit=50, seed=1)
    prefix = tasks.load_rows(tmp_path / "h50" / "manifest.json")
    assert [r["id"] for r in prefix] == [r["id"] for r in rows[:50]]

    # a pool smaller than the split size is kept whole in pool order
    # (trim_dataset returns the dataset unchanged): 500 rows -> train pool = 100 < 150
    monkeypatch.setattr(hotpotqa, "_load_train_rows", lambda: _fake_hotpot_rows(500))
    small = hotpotqa.download(tmp_path / "h-train", limit=0, seed=1, split="train")
    assert small["kept"] == 100 and small["pool"] == [400, 500]
    train_rows = tasks.load_rows(tmp_path / "h-train" / "manifest.json")
    assert [r["source_index"] for r in train_rows] == list(range(400, 500))


def test_hotpotqa_cases_score_with_squad_normalisation(common, tmp_path, monkeypatch):
    _, tasks, _, _ = common
    from _common.tasks import hotpotqa

    monkeypatch.setattr(hotpotqa, "_load_train_rows", lambda: _fake_hotpot_rows(10))
    tasks.get("hotpotqa_gepa").download(tmp_path / "h", limit=0, seed=1)  # pool of 4, kept whole
    batch, _rows = tasks.build_cases(tasks.get("hotpotqa_gepa"), tmp_path / "h" / "manifest.json")
    cases = list(batch)
    beatles = cases[1]                                            # i=1 -> "The Beatles"
    assert beatles.metadata["gold"] == "The Beatles"
    assert tasks.score_case(beatles, "Reasoning...\nAnswer: The Beatles.")
    assert tasks.score_case(beatles, "the answer is beatles")     # article + case + punctuation
    assert not tasks.score_case(beatles, "Answer: The Beatles tribute")
    assert not tasks.score_case(beatles, "The Rolling Stones")
    yes = cases[0]                                                # i=0 -> "yes"
    assert yes.metadata["gold"] == "yes"
    assert tasks.score_case(yes, "Answer: Yes.")
    assert not tasks.score_case(yes, "Answer: no")


def test_gsm8k_is_registered(common):
    _, tasks, _, _ = common
    task = tasks.get("gsm8k")
    assert task.modality == "llm" and task.kind == "exact_or_numeric"
    assert (task.default_limit, task.default_seed, task.max_new_tokens) == (500, 0, 1024)
    assert not task.short_answer
    assert "gsm8k" in tasks.names("llm")
    assert tasks.DEFAULT_TASK["llm"] == "bbh_causal_judgement"  # opt-in, not the default


def _fake_gsm8k_rows(n):
    return [
        {"question": f"Problem {i}?",
         "answer": f"Step one.\nStep <<2+2={i}>>two.\n#### " + ("1,234" if i == 3 else str(i * 3))}
        for i in range(n)
    ]


def test_gsm8k_download_samples_in_test_order_with_numeric_golds(common, tmp_path, monkeypatch):
    import random

    _, tasks, _, _ = common
    from _common.tasks import gsm8k

    monkeypatch.setattr(gsm8k, "_load_test_rows", lambda: _fake_gsm8k_rows(20))
    summary = tasks.get("gsm8k").download(tmp_path / "g", limit=10, seed=0)
    assert summary["kept"] == 10 and summary["test_rows"] == 20
    rows = tasks.load_rows(tmp_path / "g" / "manifest.json")
    expected = sorted(random.Random(0).sample(range(20), 10))
    assert [r["source_index"] for r in rows] == expected          # test-file order
    first = rows[0]
    i = expected[0]
    assert first["prompt"] == f"Problem {i}?\n\n{gsm8k.INSTRUCTION}"
    assert first["answers"] == ["1234" if i == 3 else str(i * 3)]  # comma stripped
    assert first["task"] == "exact_or_numeric" and first["numeric_tolerance"] == 0.0
    assert first["metadata"]["n_reasoning_steps"] == 2

    # limit=0 freezes the whole split; the sample is seed-deterministic
    full = tasks.get("gsm8k").download(tmp_path / "g-all", limit=0, seed=0)
    assert full["kept"] == 20
    tasks.get("gsm8k").download(tmp_path / "g2", limit=10, seed=0)
    assert [r["id"] for r in tasks.load_rows(tmp_path / "g2" / "manifest.json")] == [r["id"] for r in rows]


def test_gsm8k_cases_score_through_the_numeric_grader(common, tmp_path, monkeypatch):
    _, tasks, _, _ = common
    from _common.tasks import gsm8k

    monkeypatch.setattr(gsm8k, "_load_test_rows", lambda: _fake_gsm8k_rows(5))
    tasks.get("gsm8k").download(tmp_path / "g", limit=0, seed=0)
    batch, _rows = tasks.build_cases(tasks.get("gsm8k"), tmp_path / "g" / "manifest.json")
    cases = list(batch)
    four = cases[4]                                               # gold "12"
    assert four.metadata["gold"] == "12"
    assert tasks.score_case(four, "Reasoning...\nAnswer: 12")
    assert tasks.score_case(four, "Answer: $12.00")               # currency + decimals
    assert not tasks.score_case(four, "Answer: 13")
    comma = cases[3]                                              # gold "1234"
    assert tasks.score_case(comma, "Answer: 1,234")
    assert not tasks.score_case(comma, "Answer: 1234.5")


def test_llm_cells_default_to_the_endpoint_backend(common):
    """Text cells run against a served model unless --backend says otherwise;
    image/audio cells stay in-process; the gemini family ignores both."""
    models, _, _, run = common
    parse = run.build_parser().parse_args
    assert parse(["--modality", "llm", "--model", "qwen3.5-2b"]).backend is None  # resolved per modality
    assert models.default_backend("llm") == "endpoint"
    assert models.default_backend("vlm") == "hf_local" and models.default_backend("alm") == "hf_local"
    llm = models.resolve("qwen3.5-2b", "llm")
    assert llm.backend == "endpoint" and llm.spec_key == "qwen3.5-2b"          # no endpoint spec -> same key
    assert models.resolve("nemotron-3-nano-4b", "llm").spec_key == "nemotron-3-nano-4b-fp8"
    assert models.resolve("qwen3.5-2b", "vlm").backend == "hf_local"
    assert models.resolve("gemma-4-e2b", "alm").backend == "hf_local"
    assert models.resolve("qwen3.5-2b", "llm", "hf_local").backend == "hf_local"  # explicit flag wins
    assert models.resolve("gemini-2.5-flash-lite", "llm").backend == "gemini"     # family forced
    with pytest.raises(ValueError):
        models.default_backend("video")
