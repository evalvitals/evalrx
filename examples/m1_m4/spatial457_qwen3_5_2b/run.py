import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_benchmark_common import BenchmarkConfig, main

if __name__ == "__main__":
    main(BenchmarkConfig(
        # Qwen3.5-2B loaded with its vision tower (spec "qwen3.5-2b-vl": thinking
        # OFF on every template render; hybrid linear/full attention stack).
        model="qwen3.5-2b-vl",
        name="Spatial457/L5_6d_spatial",
        manifest="data/manifest.json",
        task_domain="6D visual spatial reasoning",
        description=(
            "Evaluate a vision-language model on Spatial457 level-5 6D spatial questions. "
            "Each synthetic scene requires grounding multiple objects and reasoning about "
            "location, orientation, depth, and object attributes. Diagnose whether errors "
            "come from answer extraction, unstable visual grounding, or a verification gap."
        ),
        success_criteria=(
            "The short answer must exactly match the official Spatial457 answer after "
            "case, punctuation, article, and yes/no normalization."
        ),
    ))

