# Colab accelerator notebooks

The [Colab notebook index](https://github.com/evalvitals/evalrx/tree/ruinan/examples/colab) links the completed TPU word-sorting example and tracks the remaining TPU/GPU notebooks awaiting completed execution.

Each notebook installs EvalRX and Antigravity CLI (`agy`), launches the coding agent in the runtime, runs `evalrx run`, and displays the baseline, agent analysis, repair selection, and independent confirmation. Save the executed notebook to keep its outputs, generated code, tables, and figures together.

Add `GEMINI_API_KEY` to Colab Secrets and enable notebook access. The notebook sets agy's `modelProvider` to `gemini`; the key alone is insufficient. No interactive login is required. See [agy authentication](https://www.antigravity.google/docs/cli/install/).

The target model runs locally: `jax_local` on TPU, `hf_local` on GPU. Agent reasoning uses the external API. The example's output identifies the actual validation hardware. Inference repairs leave weights unchanged; an EXPLORE gain is not a confirmed repair.

The planned hardware pairs use Gemma 4 E2B-it with the same two datasets: BBH word sorting and CRUXEval output prediction. Each pair keeps the 128 examples, seed, EXPLORE/CONFIRM split, generation budget, and repair settings identical, so the notebooks demonstrate hardware adaptation of the same experiment.

See the [CLI reference](cli.md#evalrx-run) and [JAX backend design](design_jax_backend.md) for supported flags and capabilities.
