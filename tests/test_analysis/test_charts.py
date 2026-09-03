"""Phase A — host-side deterministic chart rendering (render_chart_specs)."""

from __future__ import annotations

import pytest

from evalrx.viz import renderer as charts_mod
from evalrx.viz import style as style_mod
from evalrx.viz.renderer import render_chart_specs

_HAVE_MPL = charts_mod._import_matplotlib() is not None


def _write_table(d):
    (d / "tables").mkdir()
    (d / "tables" / "t.csv").write_text("grp,val\na,3\nb,7\nc,2\n", encoding="utf-8")
    return [{"name": "g1", "kind": "bar", "data": "tables/t.csv",
             "x": "grp", "y": "val", "title": "Vals by group"}]


def test_description_always_set_and_input_not_mutated(tmp_path):
    charts = _write_table(tmp_path)
    original = dict(charts[0])
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert out[0]["description"]                 # always synthesized
    assert charts[0] == original                 # caller's list untouched
    assert out is not charts


def test_empty_input_returns_empty():
    assert render_chart_specs(None, None, "/tmp/x") == []
    assert render_chart_specs([], None, "/tmp/x") == []


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_missing_csv_is_annotated_not_dropped(tmp_path):
    charts = [{"name": "m", "kind": "bar", "data": "tables/none.csv", "x": "g", "y": "v"}]
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert len(out) == 1
    assert "not found" in out[0]["render_skipped"].lower()
    assert "figure_path" not in out[0]


def test_graceful_fallback_when_matplotlib_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(charts_mod, "_import_matplotlib", lambda: None)
    charts = _write_table(tmp_path)
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert "matplotlib" in out[0]["render_skipped"]
    assert "figure_path" not in out[0]
    assert out[0]["description"]                 # text fallback still present


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_renders_png_for_each_kind(tmp_path):
    from pathlib import Path

    _write_table(tmp_path)
    specs = [
        {"name": "bar", "kind": "bar", "data": "tables/t.csv", "x": "grp", "y": "val"},
        {"name": "line", "kind": "line", "data": "tables/t.csv", "x": "grp", "y": "val"},
        {"name": "sca", "kind": "scatter", "data": "tables/t.csv", "x": "val", "y": "val"},
    ]
    out = render_chart_specs(specs, tmp_path / "tables", tmp_path / "out")
    for spec in out:
        assert Path(spec["figure_path"]).exists()
        assert "render_skipped" not in spec


def test_nature_style_loaded_from_vendored_skill():
    # The host render style is sourced from the vendored nature-figure skill:
    # the palette (blue_main) and the spines-off rcParams. The skill's palette
    # values are synced to the dataviz-validated palette, so spec PNGs share
    # one palette with agent figures and host plotly charts.
    style_mod._NATURE_STYLE_CACHE = None
    style = style_mod.load_nature_style()
    assert style["colors"][0] == "#2a78d6"            # PALETTE["blue_main"]
    assert "#0F4D92" not in style["colors"]           # pre-sync blue retired
    assert style["rc"]["axes.spines.right"] is False
    assert style["rc"]["axes.spines.top"] is False
    assert style["rc"]["legend.frameon"] is False


def test_style_falls_back_when_skill_absent(monkeypatch, tmp_path):
    style_mod._NATURE_STYLE_CACHE = None
    monkeypatch.setattr(style_mod, "NATURE_SKILL_DIR", tmp_path / "nope")
    style = style_mod.load_nature_style()
    assert style["colors"] == style_mod.NATURE_COLORS_FALLBACK
    assert style["rc"]["axes.spines.top"] is False     # fallback still nature-clean
    style_mod._NATURE_STYLE_CACHE = None               # reset for other tests


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_styled_render_still_produces_png(tmp_path):
    charts = _write_table(tmp_path)
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    from pathlib import Path
    assert Path(out[0]["figure_path"]).exists()


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_render_is_deterministic(tmp_path):
    from pathlib import Path

    charts = _write_table(tmp_path)
    a = render_chart_specs(charts, tmp_path / "tables", tmp_path / "a")[0]["figure_path"]
    b = render_chart_specs(charts, tmp_path / "tables", tmp_path / "b")[0]["figure_path"]
    # Same spec + same CSV -> byte-identical PNG (pinned metadata, no timestamp).
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_non_utf8_and_oversized_csv_never_raise(tmp_path):
    # Contract: render_chart_specs NEVER raises into the pipeline, even on a CSV
    # with non-UTF-8 bytes or a field larger than csv.field_size_limit().
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "bin.csv").write_bytes(b"grp,val\n\xff\xfe,3\nb,7\n")
    huge = "x" * 200_000
    (tmp_path / "tables" / "huge.csv").write_text(
        f"grp,val\n{huge},3\nb,7\n", encoding="utf-8"
    )
    specs = [
        {"name": "bin", "kind": "bar", "data": "tables/bin.csv", "x": "grp", "y": "val"},
        {"name": "huge", "kind": "bar", "data": "tables/huge.csv", "x": "grp", "y": "val"},
    ]
    # Must return annotated specs, not raise.
    out = render_chart_specs(specs, tmp_path / "tables", tmp_path / "out")
    assert len(out) == 2
    for spec in out:
        assert "description" in spec  # always degrades gracefully


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_unknown_x_column_skips_without_raising(tmp_path):
    _write_table(tmp_path)
    out = render_chart_specs(
        [{"name": "bad", "kind": "bar", "data": "tables/t.csv", "x": "nope", "y": "val"}],
        tmp_path / "tables", tmp_path / "out",
    )
    assert "not in table" in out[0]["render_skipped"]


# ---------------------------------------------------------------------------
# Chart FORM follows the eval-chart-style policy, not the spec's kind alone
# (audiocaps / mmau explore runs, 2026-08-20: six of eight host-rendered specs
# were bars, four of them two-bar "mean by outcome" charts the skill forbids).
# ---------------------------------------------------------------------------

def _csv(d, name, text):
    (d / "tables").mkdir(exist_ok=True)
    (d / "tables" / f"{name}.csv").write_text(text, encoding="utf-8")
    return f"tables/{name}.csv"


def _one(tmp_path, spec):
    return render_chart_specs([spec], tmp_path / "tables", tmp_path / "out")[0]


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_class_balance_renders_as_one_composition_strip(tmp_path):
    from pathlib import Path
    data = _csv(tmp_path, "class_balance", "outcome,count\nFAIL,40\nPASS,80\n")
    out = _one(tmp_path, {"name": "class_balance", "kind": "bar", "data": data,
                          "x": "outcome", "y": "count", "title": "FAIL vs PASS"})
    assert out["rendered_as"] == "composition"
    assert Path(out["figure_path"]).exists()
    assert "composition strip" in out["description"]
    assert out["axis_labels"] == {"x": "Outcome", "y": "Count"}


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_two_group_rate_with_numerator_is_dot_plus_wilson_ci(tmp_path):
    data = _csv(tmp_path, "fr", "group,fail_rate,n,n_fail\nabsent,0.69,51,35\npresent,0.07,69,5\n")
    out = _one(tmp_path, {"name": "fr", "kind": "bar", "data": data,
                          "x": "group", "y": "fail_rate", "title": "t"})
    assert out["rendered_as"] == "dot_ci"
    assert "dot + 95% CI" in out["description"]


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_explicit_ci_columns_are_used_for_any_value(tmp_path):
    data = _csv(tmp_path, "m", "outcome,mean_score,n,ci_low,ci_high\nFAIL,0.4,12,0.2,0.6\nPASS,0.7,12,0.5,0.9\n")
    out = _one(tmp_path, {"name": "m", "kind": "bar", "data": data,
                          "x": "outcome", "y": "mean_score"})
    assert out["rendered_as"] == "dot_ci"


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_a_mean_without_an_interval_is_a_lollipop_never_two_bars(tmp_path):
    # mean of per-case rates: bounded, but not a binomial proportion -> no
    # invented Wilson interval; a rate with n but no numerator -> same.
    data = _csv(tmp_path, "mb", "outcome,mean_break_rate,n\nFAIL,0.17,12\nPASS,0.0,12\n")
    out = _one(tmp_path, {"name": "mb", "kind": "bar", "data": data,
                          "x": "outcome", "y": "mean_break_rate"})
    assert out["rendered_as"] == "lollipop"
    assert "lollipop" in out["description"]


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_percent_scaled_rate_gets_wilson_from_its_numerator(tmp_path):
    data = _csv(tmp_path, "pct", "outcome,pct_with_audit,n,n_with_audit\nFAIL,80,40,32\nPASS,40,80,32\n")
    out = _one(tmp_path, {"name": "pct", "kind": "bar", "data": data,
                          "x": "outcome", "y": "pct_with_audit"})
    assert out["rendered_as"] == "dot_ci"


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_ranked_effects_become_a_forest_and_count_bars_stay_bars(tmp_path):
    # Bars are for counts only: a ranked effect size over many groups is
    # demoted to a forest (horizontal dot) plot and says so.
    ranked = _csv(tmp_path, "ranked", "signal,separation\na,0.9\nb,0.7\nc,0.5\nd,0.2\ne,0.1\n")
    out = _one(tmp_path, {"name": "ranked", "kind": "bar", "data": ranked, "x": "signal", "y": "separation"})
    assert out["rendered_as"] == "forest" and out["kind"] == "forest"
    assert "demoted to forest" in out["render_note"]
    assert "forest plot" in out["description"]
    # ... and a rate over many bins becomes a line, while a count over many
    # categories stays a bar.
    binned = _csv(tmp_path, "binned", "bin,fail_rate,n\n0-20,0.6,10\n20-40,0.4,12\n40-60,0.3,9\n60-80,0.2,11\n80-100,0.1,8\n")
    out = _one(tmp_path, {"name": "binned", "kind": "bar", "data": binned, "x": "bin", "y": "fail_rate"})
    assert out["rendered_as"] == "line" and "demoted to line" in out["render_note"]
    counts = _csv(tmp_path, "vc", "category,count\nx,3\ny,7\nz,2\nw,9\nv,1\nu,4\nt,2\n")
    out = _one(tmp_path, {"name": "vc", "kind": "bar", "data": counts, "x": "category", "y": "count"})
    assert out["rendered_as"] == "bar"          # 7 classes: too many for a strip


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_line_and_scatter_keep_their_kind(tmp_path):
    data = _csv(tmp_path, "bins", "bin,fail_rate,n\n0-20,0.6,10\n20-40,0.4,12\n40-60,0.1,9\n")
    out = _one(tmp_path, {"name": "bins", "kind": "line", "data": data, "x": "bin", "y": "fail_rate"})
    assert out["rendered_as"] == "line"
    out = _one(tmp_path, {"name": "sc", "kind": "scatter", "data": data, "x": "fail_rate", "y": "n"})
    assert out["rendered_as"] == "scatter"


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_dot_ci_render_is_deterministic(tmp_path):
    from pathlib import Path
    data = _csv(tmp_path, "fr", "group,fail_rate,n,n_fail\nFAIL,0.69,51,35\nPASS,0.07,69,5\n")
    spec = {"name": "fr", "kind": "bar", "data": data, "x": "group", "y": "fail_rate"}
    a = render_chart_specs([spec], tmp_path / "tables", tmp_path / "a")[0]["figure_path"]
    b = render_chart_specs([spec], tmp_path / "tables", tmp_path / "b")[0]["figure_path"]
    assert Path(a).read_bytes() == Path(b).read_bytes()


def test_wilson_interval_matches_reference_values():
    lo, hi = charts_mod._wilson(5, 52)
    assert 0.04 < lo < 0.045 and 0.20 < hi < 0.21       # 5/52 -> [0.042, 0.207]
    assert charts_mod._wilson(0, 10)[0] == 0.0
    assert charts_mod._wilson(10, 10)[1] == pytest.approx(1.0)


def test_semantic_palette_stays_in_sync_with_eval_viz_theme():
    pytest.importorskip("plotly")
    from evalrx.analysis import eval_viz_theme as viz
    for key in ("FAIL", "PASS", "INCONCLUSIVE", "ACCENT", "LEAKY", "AXIS", "GRID", "TEXT"):
        assert style_mod.SEMANTIC_PALETTE[key] == viz._LIGHT[key], key
    assert style_mod.outcome_color("fail") == viz.OUTCOME_COLORS["FAIL"]
    assert style_mod.outcome_color("Pass") == viz.OUTCOME_COLORS["PASS"]
    assert style_mod.outcome_color("absent") is None
