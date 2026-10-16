# Contributing to EvalRX

## Setup

```bash
git clone https://github.com/evalvitals/evalrx.git
cd evalrx
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch for tests
pip install -e ".[dev,local,api,data,finetune,viz,ui,contract]"
```

## Running tests

```bash
pytest -m "not gpu and not network"  # same selection as CI
pytest --run-gpu      # GPU integration tests (requires CUDA + cached weights)
```

The report tests need the built frontend. Use Node.js 22 and install from its
lockfile before building, including after a branch change updates dependencies:

```bash
cd evalrx/reporting/web
npm ci
npm run typecheck
npm run build
```

From the repository root, validate the generated contracts and documentation:

```bash
python -m evalrx.contract.export --out docs/contract --check
pip install mkdocs-material
mkdocs build --strict
```

Keep run data, local notebooks, and standalone report exports out of commits.
The [local artifact policy](docs/local_artifacts.md) lists retained paths and
provides an inventory command. Generated files are preserved locally.

## Linting

```bash
ruff check .
mypy evalrx --ignore-missing-imports
```

## Adding a model

1. Add a `ModelSpec` entry in `evalrx/specs.py` (`_add(ModelSpec(key=..., ...))`).
2. If white-box internals are needed, add `evalrx/models/whitebox/<family>.py` — declare `capabilities` + implement `forward(inputs, capture) -> Trace` — and register with `@register_model`.
3. The `compose()` function in `evalrx/models/compose.py` wires everything together; no changes needed there for a new spec-only model.
4. Write a unit test in `tests/test_models/` using `FakeModel` from `tests/conftest.py` (no GPU needed).

See `docs/extending.md` for a full walkthrough.

## Adding an analyzer

1. Create `evalrx/analyzers/<category>/<name>.py`.
2. Declare `name`, `requires` (frozenset of `Capability`), and `applies_to_modalities`.
3. Implement `_run(self, model, cases: CaseBatch) -> Result`.
4. Decorate with `@register_analyzer("<name>")`.
5. Import the new module in `evalrx/analyzers/<category>/__init__.py`.
6. Write a unit test using `FakeModel`.

The new analyzer is instantly discoverable via `registry.analyzers.names_compatible_with(model)` — no other changes needed.

## Pull requests

- Keep PRs focused: one analyzer, one model, or one infrastructure change per PR.
- All configured CI checks must pass: Ruff, the frontend build, contract export,
  documentation build, and the non-GPU/non-network test suite. Mypy is an
  additional local check, not currently a CI gate.
- Add or update the relevant entry in `CHANGELOG.md` under `[Unreleased]`.
- Include paper citations (arXiv link + BibTeX key) for any method from the literature.
