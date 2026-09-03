from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "examples/m1_m4/vlm_benchmark_common.py"
    spec = importlib.util.spec_from_file_location("vlm_benchmark_common_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fix_scorer_extracts_final_answer_instead_of_first_reasoning_line():
    common = _module()
    output = "Values: 40 and 2\nComputation: 40 + 2\nFinal answer: 42"
    assert common.answer_matches(output, ["42"], numeric_tolerance=0.05)
    assert not common.answer_matches(output, ["39"], numeric_tolerance=0.05)


def test_chartqa_numeric_tolerance_and_percent_format():
    common = _module()
    assert common.answer_matches("Final answer: 6.8%", ["6.8"], numeric_tolerance=0.05)
    assert common.answer_matches("105", ["100"], numeric_tolerance=0.05)
    assert not common.answer_matches("106", ["100"], numeric_tolerance=0.05)


def test_examples_choose_their_model_through_the_config():
    """BenchmarkConfig.model is the --model default: the Qwen2.5-VL examples keep
    the 7B key, the Qwen3.5 examples name the vision-tower spec."""
    import re
    from pathlib import Path

    from evalrx.models import resolve_spec_key

    common = _module()
    assert common.BenchmarkConfig.model == "qwen2.5-vl-7b-instruct"  # the default
    root = Path(__file__).parents[2] / "examples/m1_m4"
    for example in ("chartqa_qwen3_5_2b", "spatial457_qwen3_5_2b"):
        src = (root / example / "run.py").read_text()
        key = re.search(r'model="([^"]+)"', src).group(1)
        assert key == "qwen3.5-2b-vl"
        assert resolve_spec_key(key) == key  # a registered spec, not an alias
    for example in ("chartqa_qwen2_5_vl", "spatial457_qwen2_5_vl"):
        assert 'model=' not in (root / example / "run.py").read_text()
