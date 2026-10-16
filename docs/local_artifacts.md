# Local artifacts and source files

Experimental data and results remain at their original paths. Git ignores
generated files so that source changes can be reviewed without adding entire
run directories. Ignoring a file does not delete it or back it up.

## What belongs in Git

- Library code, tests, configuration, scripts, and runnable example instructions.
- Curated validation reports with explicit model, protocol, date, and limitations.
- Small reference fixtures and frozen manifests already used by examples/tests.
- The published TPU notebook, including its reviewed execution evidence.

Raw logs, downloaded media, temporary workspaces, incomplete notebook executions,
and generated standalone reports remain local. New results should include a
short source-controlled report and point to the retained evidence by repository-
relative path. Do not bulk-add an experiment directory.

## Retained artifact index

Paths below are relative to the repository root. They are filesystem locations,
not download links in the documentation website.

| Location | Contents and status | Version-control policy |
| --- | --- | --- |
| `examples/benchmark/{llm,vlm,alm}/_data/`, `examples/benchmark/**/host_data/` | Downloaded media and frozen input data shared across model families | Local, ignored; preserve input provenance |
| `examples/benchmark/**/host_outputs/`, `examples/**/outputs*` | Historical predictions, logs, and run reports | Local, ignored; retain completed and failed attempts |
| `examples/benchmark/smoke_checks/api-smoke-20261012/` | Gemini API smoke evidence | Local, ignored |
| `examples/benchmark/smoke_checks/m1m5-repair-20261012/` | Earlier repair attempts | Local, ignored; not a substitute for final confirmation |
| `examples/benchmark/smoke_checks/gemma-model-swap-20261012/` | Model-swap diagnostic artifacts | Local, ignored |
| `examples/benchmark/smoke_checks/agentic-default-20261012/` | Frozen Gemini M1–M5 evidence, failed initial BBH attempt, retries and recovered workspaces | `README.md` and `audit.py` are source; JSON/logs/media/workspaces stay local |
| `examples/m1_m4/**/data/` | Historical input manifests and media retained after example reorganization | Local, ignored; runnable examples are indexed in `examples/README.md` |
| `examples/colab/gpu/code_reasoning_repair.ipynb`, `examples/colab/gpu/word_sorting_repair.ipynb` | Incomplete local notebook executions | Local, ignored until separately validated for publication; outputs preserved |
| `examples/m1_m5/mmau_qwen2_audio/demo_page/index.html` | 91.2 MiB historical standalone demo | Local, ignored; recover with `python tools/restore_mmau_demo.py` |
| `build/`, `site/`, `evalrx/reporting/web_dist/`, `evalrx/reporting/web/node_modules/` | Local builds and installed frontend packages | Ignored; regenerate from source and the frontend lockfile |
| `.local-artifacts/` | Local inventories and maintenance verification records | Ignored; may contain machine-specific paths and timestamps |

The repository also keeps curated website demos under `docs/demo/`. These are
published site assets, separate from per-run standalone exports.

## Inventory and preservation checks

Create a file-level inventory of ignored experiment artifacts, including sizes
and SHA-256 digests, without changing any input or result:

```bash
python tools/inventory_local_artifacts.py --hash
```

The output is `.local-artifacts/inventory.json`. It records repository-relative
paths, the source revision, file counts, bytes and hashes. Files changing during
the scan cause a nonzero exit so the result is not mistaken for a stable snapshot.
Without `--hash`, the command records only sizes and modification times.
Use `--output` to save a dated inventory before and after maintenance.

Keep separate backups of valuable results. Do not use `git clean -fdx` as a
cleanup procedure: it would remove the ignored evidence listed above.

## Validation records

- [Gemini agentic validation, 2026-10-13](gemini_agentic_report_20261013.md).
- [Gemma repair matrix, historical snapshot from 2026-08-30](validation/gemma_repair_20260830.md).

These reports describe their recorded runs; repository maintenance does not
rerun the models or change the reported metrics.
