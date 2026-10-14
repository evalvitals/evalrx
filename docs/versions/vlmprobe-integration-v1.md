# VLMProbe integration v1

## Goal

Make the 24 public VLMProbe factories available through EvalRX's analyzer
contract, while keeping method execution, model adapters, calibrated artifacts
and built-in visualizations in the maintained VLMProbe package.

## Differences from the previous EvalRX version

- Adds one registered `vlmprobe` analyzer with explicit `family.factory`
  selection instead of duplicating 24 scientific kernel implementations.
- Executes against the same loaded local PyTorch model and native processed
  inputs. Gradient and causal methods use the live computation, not detached
  observational traces.
- Retains native `ProbeResult` tensors and provenance alongside compact
  EvalRX findings, with visualization and portable artifact persistence.
- Supports explicit reference/target inputs, path sweeps, GPT-2 circuit
  recovery, and joint steering generation where the method supports them.
- Loads calibrated lens artifacts or accepts explicitly fitted methods.
  Training is performed separately through the bound method, not on evaluation
  cases during analyzer execution.
- Adds an optional `vlmprobe` installation extra, a complete method/scope guide,
  and an offline CPU example. The core install stays Torch-free; EvalRX's
  existing analyzer names and local-model version constraints remain intact.

## Expected experimental results

The bridge should reproduce the underlying VLMProbe method's tensors and
score changes on identical native inputs/configurations. Self-replacement
and zero-strength controls should produce zero effects. Unsupported adapter
capabilities and uncalibrated lenses should fail with actionable explanations.
Saved artifacts should preserve method configuration, coordinates and scores;
visualizations should use these measured tensors without invented labels.

The offline tiny random GPT-2 example tests API behavior and controls only.
It is not expected to exhibit meaningful learned language or vision behavior,
and no pretrained model, dataset-scale finding, or new training result is
claimed by this integration version.

## Validation

The offline CPU example completed with Transformers 5.3.0 and local VLMProbe
0.9.0, without downloading models/data. It produced `[2,4,12]` lens logits,
actual and estimated patch effects, portable artifacts and a lens heatmap;
the two self-replacement effects were exactly zero. The method guide covers
all 24 live public factories, and the generated analyzer index was refreshed.

Validation against the published PyPI `vlmprobe==0.9.0` wheel completed with
**96 integration tests passed and one intentional architecture skip** (the
native GPT-2 transformer graph is unavailable on Llama). Tests include tiny
GPT-2, Llama, LLaVA and ViT models built locally without downloaded weights;
all 24 factory dispatch routes; actual gradients, reference replacements,
input-path maps, VSV and genuine ViT relevance propagation; and calibration,
save/load and identity checks for all three fitted lenses.

The original core, model-wrap, analyzer-contract and analyzer-documentation
regression suite passed **714 tests**. Whole-repository Ruff lint, generated
analyzer-documentation checks, `git diff --check`, and wheel construction also
passed. The wheel contains all five integration modules and declares VLMProbe
only as an optional dependency. Import checks block Torch, Transformers and
VLMProbe to verify that the lightweight EvalRX core still imports.

The integration retains upstream capability and numerical guards. In 0.9.0,
GPT-2 does not expose the editable attention needed by reweighting, temperature
and knockout. Temperature sharpening at 0.7 on finite-minimum masked Llama
scores triggers the upstream overflow guard; successful temperature parity
tests use 1.5. This integration does not bypass those guards or alter method
kernels. The dedicated CI workflow uses the published package and audited
Transformers 5.3 environment.

The [GitHub Actions integration run](https://github.com/evalvitals/evalrx/actions/runs/38530824251)
passed in a fresh CPU environment, including the 96 passing tests, intentional
architecture skip and offline example. The follow-up documentation clarifies
that users must install the `gexin` checkout until EvalRX publishes this extra.
