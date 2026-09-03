# Contributing to EvalRX

## Setup

```bash
git clone https://github.com/evalvitals/evalrx.git
cd evalrx
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch for tests
```

## Running tests

```bash
pytest -m "not gpu"   # fast unit tests, no GPU needed (~165 tests)
pytest --run-gpu      # GPU integration tests (requires CUDA + cached weights)
```

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
- All CI checks (ruff, mypy, pytest fast suite) must pass.
- Add or update the relevant entry in `CHANGELOG.md` under `[Unreleased]`.
- Include paper citations (arXiv link + BibTeX key) for any method from the literature.
