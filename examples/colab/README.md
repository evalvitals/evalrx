# Colab notebooks

Select the matching hardware runtime, add `GEMINI_API_KEY` to Colab Secrets, then run the notebook from top to bottom. The notebooks install **Antigravity CLI (`agy`)** and configure API-key authentication; no browser login is needed. Agent code, charts, predictions, and confirmation results remain in the notebook outputs.

| Hardware | Model | Task | Notebook |
|---|---|---|---|
| TPU | Gemma 4 E2B-it | BBH word sorting | [Word sorting](tpu/word_sorting_repair.ipynb) |
| TPU | Gemma 4 E2B-it | CRUXEval output prediction | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | BBH word sorting | Execution in progress; notebook pending |
| GPU | Gemma 4 E2B-it | CRUXEval output prediction | Queued; notebook pending |

The planned TPU/GPU pairs use the same model and dataset, 128 cases, seed 0, a 64/64 EXPLORE/CONFIRM split, a 512-token generation budget, and the same agent and repair settings. The hardware adaptation changes installation, device checks, and `--backend`/`--device`: `jax_local`/`tpu` versus `hf_local`/`cuda`. Both default to bfloat16; use a compatible GPU such as L4 or A100. Backend numerics can still produce different predictions.

The executed TPU word-sorting notebook reports CONFIRM accuracy of **27/64 → 37/64** (13 fixed, 3 broken). Its e-value is **6.884**, below the threshold of 20: the verdict is **partial**, not a validated repair. All six code cells completed and their outputs are saved. The remaining three notebooks will be published after execution.

A successful repair must pass the independent CONFIRM check. EXPLORE gains alone are not enough. These are inference repairs around unchanged model weights, not fine-tuning.

The earlier [ChartQA diagnosis](archive/chartqa_diagnosis.ipynb) is retained with its original outputs and its negative result: no validated repair.

Google recommends migrating individual-account Gemini CLI users to agy; API-key Gemini CLI access remains supported. agy API-key authentication requires `modelProvider: "gemini"` in its settings as well as `GEMINI_API_KEY`. See the [official migration announcement](https://github.com/google-gemini/gemini-cli/discussions/28017) and [authentication instructions](https://www.antigravity.google/docs/cli/install/).
