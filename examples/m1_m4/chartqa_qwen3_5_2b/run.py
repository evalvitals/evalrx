import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_benchmark_common import BenchmarkConfig, main

if __name__ == "__main__":
    main(BenchmarkConfig(
        # Qwen3.5-2B loaded with its vision tower (spec "qwen3.5-2b-vl": thinking
        # OFF on every template render; hybrid linear/full attention stack).
        model="qwen3.5-2b-vl",
        name="ChartQA/test_human",
        manifest="data/manifest.json",
        task_domain="chart visual question answering",
        description=(
            "Evaluate a vision-language model on the human-authored ChartQA test split. "
            "Questions require reading chart marks and labels, comparing quantities, and "
            "performing arithmetic or logical reasoning. Diagnose whether errors reflect "
            "answer extraction, unstable visual evidence, or failure to verify a candidate answer."
        ),
        success_criteria=(
            "The short answer must match an official ChartQA label after normalization; "
            "numeric answers use ChartQA's relaxed five-percent tolerance."
        ),
    ))

