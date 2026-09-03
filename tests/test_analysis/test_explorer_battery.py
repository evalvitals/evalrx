"""The strengthened explorer prompt asks for a rich chart battery (Lambda-style).

We can't unit-test what a live LLM emits, but we CAN prove the contract: when the
agent writes several CSV tables + a list of chart specs, the explorer surfaces
them all and the host renders each one. This guards the explorer->render chain
against regressions that would silently drop charts.
"""

from __future__ import annotations

from pathlib import Path

from evalrx.analysis.explorer import ExploratoryAnalysisAgent
from evalrx.viz.renderer import render_chart_specs

# A compliant "Lambda-style" script: pre-aggregated CSVs + one spec per CSV.
_RICH_SCRIPT = r'''
import json, csv, os, collections
rows = json.load(open("records.json"))
os.makedirs("tables", exist_ok=True)
def is_fail(r): return 1 if str(r.get("label","")).lower() in ("fail","0","false") else 0

nb = {"FAIL": sum(is_fail(r) for r in rows), "PASS": sum(1 - is_fail(r) for r in rows)}
with open("tables/class_balance.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["outcome","count"]); [w.writerow([k,v]) for k,v in nb.items()]

bins = collections.defaultdict(lambda:[0,0])
for r in rows:
    b = "small" if float(r.get("obj_size",0)) < 40 else "large"
    bins[b][0]+=is_fail(r); bins[b][1]+=1
with open("tables/failrate_by_size.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["obj_size_bin","fail_rate"])
    [w.writerow([b, round(c[0]/max(1,c[1]),3)]) for b,c in bins.items()]

with open("tables/top_discriminators.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["signal","separation"]); w.writerow(["obj_size",0.8]); w.writerow(["attention",0.4])

charts = [
 {"name":"class_balance","kind":"bar","data":"tables/class_balance.csv","x":"outcome","y":"count","title":"FAIL vs PASS"},
 {"name":"failrate_by_size","kind":"line","data":"tables/failrate_by_size.csv","x":"obj_size_bin","y":"fail_rate","title":"Fail rate by size"},
 {"name":"top_discriminators","kind":"bar","data":"tables/top_discriminators.csv","x":"signal","y":"separation","title":"Top discriminators"},
]
visual_plan = [
 {"name":"class_balance","question":"How many FAIL/PASS cases exist?","data_shape":"categorical-vs-count",
  "plot_kind":"bar","fallback_kind":"bar","required_columns":["label"],
  "rationale":"Counts by class are categorical totals."},
 {"name":"failrate_by_size","question":"Does size change failure risk?","data_shape":"numeric-vs-binary",
  "plot_kind":"line","fallback_kind":"line","required_columns":["obj_size","label"],
  "rationale":"Ordered bins show a risk curve."},
 {"name":"top_discriminators","question":"Which signal separates FAIL/PASS most?","data_shape":"many-numeric",
  "plot_kind":"bar","fallback_kind":"bar","required_columns":["obj_size","attention","label"],
  "rationale":"A ranked bar compares signal effect magnitudes."},
]
print("EXPLORATORY_RESULT_JSON=" + json.dumps({
  "observations":["FAIL skews small"],
  "visual_plan": visual_plan,
  "candidate_signals":[{"name":"small","rationale":"r","recipe":{"name":"small","kind":"expr","expr":"obj_size < 40"}}],
  "plots":[], "tables":{}, "charts":charts, "caveats":["exploratory only"],
  "recommended_confirmatory_tests":["confirm small"]}))
'''


class _FakeJudge:
    def generate(self, prompt, **kw):
        return "```python\n" + _RICH_SCRIPT + "\n```"


def _records():
    return [
        {"obj_size": 20, "attention": 0.1, "label": "fail"},
        {"obj_size": 80, "attention": 0.6, "label": "pass"},
        {"obj_size": 25, "attention": 0.2, "label": "fail"},
        {"obj_size": 90, "attention": 0.7, "label": "pass"},
    ]


def test_explorer_surfaces_multi_chart_battery():
    agent = ExploratoryAnalysisAgent(judge=_FakeJudge(), max_attempts=1)
    rep = agent.explore_records(_records(), question="distinguish FAIL from PASS")
    assert rep.ok
    names = [c["name"] for c in rep.charts]
    assert names == ["class_balance", "failrate_by_size", "top_discriminators"]
    assert [v["name"] for v in rep.visual_plan] == names
    assert rep.visual_plan[1]["plot_kind"] == "line"
    # each chart references a CSV the script wrote
    for c in rep.charts:
        assert c["data"].startswith("tables/")


def test_battery_charts_all_render_to_png():
    agent = ExploratoryAnalysisAgent(judge=_FakeJudge(), max_attempts=1)
    rep = agent.explore_records(_records(), question="q")
    out = Path(rep.workdir)
    rendered = render_chart_specs(rep.charts, out, out)
    pngs = [c for c in rendered if c.get("figure_path") and Path(c["figure_path"]).exists()]
    # matplotlib is a test dep via the viz extra; if absent, charts degrade gracefully
    import importlib.util
    if importlib.util.find_spec("matplotlib"):
        assert len(pngs) == 3
    else:
        assert all("render_skipped" in c for c in rendered)
