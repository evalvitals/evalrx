"""Offline CPU example: VLMProbe readouts and replacement through EvalRX.

Install in the audited probing environment:
    pip install "evalrx[local,vlmprobe]" "transformers>=5.3,<5.4"
Run from the repository root:
    python examples/vlmprobe_integration.py --output-dir /tmp/evalrx-vlmprobe

This example constructs RANDOM GPT-2 weights and an in-memory tokenizer. It
downloads no weights or data. Its outputs verify the integration contract;
they are not evidence about a pretrained model's language understanding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vlmprobe-offline"))
    args = parser.parse_args()

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    import evalrx
    from evalrx.analyzers.vlmprobe import VLMProbeAnalyzer
    from evalrx.core.case import Inputs
    from vlmprobe import Prober

    torch.manual_seed(7)
    torch.set_num_threads(1)
    vocabulary = {
        "[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
        "The": 3, "shirt": 4, "is": 5, "yellow": 6, "black": 7,
        "What": 8, "color": 9, "?": 10, ".": 11,
    }
    backend_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
    )
    configuration = GPT2Config(
        vocab_size=len(vocabulary), n_positions=32, n_ctx=32,
        n_embd=16, n_layer=2, n_head=2,
        bos_token_id=2, eos_token_id=2, pad_token_id=1,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    configuration._attn_implementation = "eager"
    torch_model = GPT2LMHeadModel(configuration).eval()
    model = evalrx.wrap(torch_model, tokenizer)
    probe = Prober(torch_model, processor=tokenizer)

    def native_inputs(_model, case):
        return tokenizer(case.inputs.prompt, return_tensors="pt")

    case = evalrx.FailureCase(
        id="offline-random-gpt2", inputs=Inputs(prompt="The shirt is yellow"),
    )
    common = {"prober": probe, "input_builder": native_inputs}
    metric = {
        "kind": "token_margin", "positive": vocabulary["yellow"],
        "negative": vocabulary["black"], "position": "last_prompt",
    }

    readout = VLMProbeAnalyzer(
        method="lens.logit", method_config={"tokens": "all"}, **common,
    ).run(model, [case])
    reference = tokenizer("The shirt is black", return_tensors="pt")
    replacement = VLMProbeAnalyzer(
        method="causal.patch", method_config={"tokens": "all"},
        run_config={"source": reference, "alignment": "position"},
        metric=metric, **common,
    ).run(model, [case])
    estimate = VLMProbeAnalyzer(
        method="causal.attribute", method_config={"tokens": "all"},
        run_config={"source": reference, "alignment": "position"},
        metric=metric, **common,
    ).run(model, [case])

    # Replacing a sequence's states with its own states is a meaningful control.
    control = VLMProbeAnalyzer(
        method="causal.patch", method_config={"tokens": "all"},
        run_config=lambda current: {"source": native_inputs(model, current)},
        metric=metric, **common,
    ).run(model, [case])
    control_effect = control.artifacts["probe_results"][0].tensors["effect"]
    torch.testing.assert_close(control_effect, torch.zeros_like(control_effect))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, result in (
        ("logit-lens", readout), ("activation-patching", replacement),
        ("attribution-patching", estimate), ("self-replacement-control", control),
    ):
        result.save_probes(args.output_dir / label)
        (args.output_dir / f"{label}.json").write_text(
            result.to_json(indent=2) + "\n", encoding="utf-8",
        )
    ax = readout.visualize(plot="lens_heatmap", target_token_id=vocabulary["yellow"])
    ax.figure.savefig(args.output_dir / "logit-lens.png", dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(ax.figure)

    print(json.dumps({
        "model": "random tiny GPT-2; integration verification only",
        "network_access": False,
        "outputs": str(args.output_dir.resolve()),
        "lens_logits_shape": list(readout.artifacts["probe_results"][0].tensors["logits"].shape),
        "activation_effect": replacement.artifacts["probe_results"][0].tensors["effect"].tolist(),
        "estimated_effect": estimate.artifacts["probe_results"][0].tensors["estimated_effect"].tolist(),
        "self_replacement_effect": control_effect.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
