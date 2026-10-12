"""docs/analyzers.md must stay in step with the registries (tools/gen_analyzer_docs.py).

The first test is the drift guard: an analyzer or model spec missing from the
page, or a registry-derived column that no longer matches the code, fails here
with the regenerate command.  The rest pin the generator's own contract on small
synthetic pages.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")  # without torch the attention analyzers do not register
pytest.importorskip("numpy")

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "gen_analyzer_docs.py"

TABLE_HEAD = (
    "| Analyzer | Key | Requires | Modality | Paper | Status |\n"
    "|---|---|---|---|---|---|\n"
)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_analyzer_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patching_page(rows: str) -> str:
    return (
        "### Patching\n\n"
        "<!-- BEGIN GENERATED: analyzers/patching -->\n"
        + TABLE_HEAD + rows
        + "<!-- END GENERATED -->\n"
    )


def test_analyzer_doc_is_current(gen):
    text = gen.DOC.read_text(encoding="utf-8")
    fresh, added = gen.render(text)
    assert not added, f"registered but not on the page: {added}"
    assert fresh == text, f"{gen.DOC} is stale -- regenerate with: {gen.REGEN_CMD}"


def test_curated_cells_survive_and_derived_cells_are_rewritten(gen):
    page = _patching_page("| My trace | `causal_trace` | WRONG | WRONG | [P](https://x) | ✓ |\n")
    fresh, added = gen.render(page, require_all=False)
    assert (
        "| My trace | `causal_trace` | `HIDDEN_STATES` | text + image | [P](https://x) | stub |"
        in fresh
    )
    assert added == []


def test_new_key_gets_a_placeholder_row_and_a_stale_key_is_dropped(gen):
    page = _patching_page("| Gone | `no_such_analyzer` | | | — | |\n")
    fresh, added = gen.render(page, require_all=False)
    assert "no_such_analyzer" not in fresh
    assert "`causal_trace`" in fresh
    assert added == ["analyzers/patching: causal_trace"]


def test_registered_package_without_a_section_is_an_error(gen):
    with pytest.raises(ValueError, match="analyzers/perturbation"):
        gen.render(_patching_page(""))


def test_check_mode_fails_on_a_stale_page_until_regenerated(gen, tmp_path, capsys):
    doc = tmp_path / "analyzers.md"
    page = gen.DOC.read_text(encoding="utf-8")
    stale = page.replace("| `rise` | `GENERATE` |", "| `rise` | `LOGITS` |")
    assert stale != page
    doc.write_text(stale)
    assert gen.main(["--doc", str(doc), "--check"]) == 1
    assert gen.REGEN_CMD in capsys.readouterr().err
    assert gen.main(["--doc", str(doc)]) == 0
    assert gen.main(["--doc", str(doc), "--check"]) == 0
