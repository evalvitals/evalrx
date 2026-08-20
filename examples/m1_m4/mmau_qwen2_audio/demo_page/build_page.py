#!/usr/bin/env python3
"""Render one finished run of this example as a single self-contained HTML page.

A one-off demo artifact, not part of the product: `evalvitals dashboard` is the
supported way to browse a run. This script exists because a demo sometimes needs
a page you can send someone — no Streamlit, no install, no `./data` on their
machine — covering the whole loop rather than only the case book.

It reads a finished run directory and the benchmark manifest beside it, and
writes ONE html file with every figure and audio clip inlined as a data URI.
The page is a snapshot: re-run this script to reflect a newer run.

    python demo_page/build_page.py \
        --run-dir outputs \
        --example-dir . \
        --out demo_page/index.html

`--run-dir` accepts either the run directory itself (the one holding
`run_log.jsonl`) or a parent holding `logs/` + `explore/`, which is how
`run.py` lays it out.

Needs `ffmpeg` on PATH to compress the clips; without it, pass `--no-audio` and
the case book renders without players.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Bitrate/samplerate for the inlined clips. MMAU clips are speech/music
#: identification tasks, so 48k AAC mono at 16 kHz stays intelligible while
#: keeping a 67-clip page around 5 MB rather than 48 MB of source wav.
AUDIO_ARGS = ["-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k"]


def resolve_run(run_dir: Path) -> tuple[Path, Path | None]:
    """Return (logs_dir, explore_dir) for either layout `run.py` can produce."""
    if (run_dir / "run_log.jsonl").exists():
        explore = run_dir.parent / "explore"
        return run_dir, explore if explore.is_dir() else None
    logs = run_dir / "logs"
    if (logs / "run_log.jsonl").exists():
        explore = run_dir / "explore"
        return logs, explore if explore.is_dir() else None
    sys.exit(f"no run_log.jsonl under {run_dir} or {run_dir / 'logs'}")


def load_manifest(example_dir: Path) -> dict:
    path = example_dir / "data" / "mmau_test_mini.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run download_mmau.py first, or pass --example-dir")
    return {json.loads(l)["id"]: json.loads(l) for l in path.open()}


def newest_fix_dir(logs: Path) -> Path | None:
    """The fix attempt the run confirmed — the one with per-case outputs."""
    cands = sorted(p for p in (logs / "fixes").glob("*/result.json"))
    if not cands:
        return None
    # Prefer an attempt that actually validated; otherwise the last one tried.
    for p in cands:
        if json.loads(p.read_text()).get("fixed"):
            return p.parent
    return cands[-1].parent


#: What each analyzer is actually asking, in a reader's words, and which
#: finding to surface as its headline. The question is editorial (the analyzers
#: don't ship a lay description); the number is read straight from the result
#: JSON, so it stays honest if the run changes.
#:   name -> (question, [(label, dotted-path-into-findings, format)])
M1_QUESTIONS: dict[str, tuple[str, list]] = {
    "answer_extraction_audit": (
        "Is a case marked wrong really wrong, or did our parser just fail to find the answer?",
        [("parse failures among wrong answers", "suspect_rate", "pct"),
         ("answered without the requested tag", "missing_tag_rate", "pct")]),
    "termination_audit": (
        "Did the output get cut off before the model finished?",
        [("outputs that look truncated", "truncation_rate", "pct"),
         ("recover an answer when allowed to continue", "recovered_rate", "pct")]),
    "calibration": (
        "When the model sounds confident, is it actually right?",
        [("calibration error, log-prob confidence", "logprob_channel.ece", "num"),
         ("calibration error, stated confidence", "verbalized_channel.ece", "num")]),
    "format_sensitivity": (
        "Reorder the A/B/C/D options — does the answer follow the content, or the letter?",
        [("answers that change under reordering", "mean_flip_rate", "pct")]),
    "coverage_verification_gap": (
        "Answer the same question 5 times — is the right answer ever among them?",
        [("cases where all 5 tries miss", "no_coverage_rate", "pct"),
         ("at least one try correct", "mean_pass_at_k", "pct")]),
    "self_consistency": (
        "Sample the same question repeatedly — does the answer wobble?",
        [("agreement across 5 samples", "consistency", "pct")]),
    "logprob_entropy": (
        "How uncertain is the model about the answer token itself?",
        [("mean top-token entropy", "mean_top_entropy", "num"),
         ("perplexity", "perplexity", "num")]),
    "selfcheck_consistency": (
        "Resample and check whether the model contradicts its own claims.",
        [("mean inconsistency", "mean_inconsistency", "num")]),
}


def _dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def read_analyzer_results(logs: Path, analyzers: list) -> list:
    """One row per analyzer: its question, how many cases it measured, headline numbers."""
    rows = []
    for name in analyzers:
        p = logs / "artifacts" / f"c0_{name}.result.json"
        findings, n = {}, None
        if p.exists():
            raw = json.loads(p.read_text())
            findings = raw.get("findings") or {}
            n = findings.get("n_cases") or findings.get("n_scored")
        question, specs = M1_QUESTIONS.get(name, ("", []))
        headline = []
        for label, path, fmt in specs:
            v = _dig(findings, path)
            if isinstance(v, (int, float)):
                headline.append({"label": label,
                                 "value": f"{v:.1%}" if fmt == "pct" else f"{v:.3f}".rstrip("0").rstrip(".")})
            else:
                headline.append({"label": label, "value": None})
        rows.append({"name": name, "question": question, "n": n, "headline": headline})
    # measured-most first; analyzers that scored nothing sink to the bottom
    rows.sort(key=lambda r: -(r["n"] or 0))
    return rows


def collect(logs: Path, explore_dir: Path | None, manifest: dict) -> dict:
    """Assemble every number the page renders into one dict."""
    rows = [json.loads(l) for l in (logs / "run_log.jsonl").open()]
    by = lambda ev: [r for r in rows if r.get("event") == ev]
    if not by("run_start"):
        sys.exit("run_log.jsonl has no run_start event")

    ex = json.loads((explore_dir / "exploratory_report.json").read_text()) if explore_dir else {}
    fixdir = newest_fix_dir(logs)
    if fixdir is None:
        sys.exit(
            f"no fix attempt under {logs / 'fixes'}\n"
            "The page renders M4 from a persisted repair attempt, which only a full run\n"
            "writes — `run.py --smoke-test` exercises the loop in-process and does not.\n"
            "Use a run produced by `docker compose up` (or `run.py --model ... --limit N`)."
        )
    res = json.loads((fixdir / "result.json").read_text())

    d: dict = {}
    rs, le = by("run_start")[0], (by("loop_end") or [{}])[0]
    d["run"] = {k: rs.get(k) for k in
                ("model", "n_cases", "label_distribution", "data_fingerprint", "evalvitals_version")}
    d["run"]["protocol"] = (rs.get("protocol") or {}).get("description", "")
    d["run"]["loop_end"] = {k: le.get(k) for k in
                            ("cycles", "stopped_by", "n_hypotheses", "n_verified", "total_duration_sec")}
    d["run"]["config"] = json.loads((logs / "manifest.json").read_text()).get("config", {})

    # --- M1 -----------------------------------------------------------------
    p0 = by("probe")[0]
    d["m1"] = {"analyzers": p0["analyzers"], "rationale": p0.get("selection_rationale"),
               "duration": p0.get("duration_sec")}
    prof = (ex.get("data_profile") or {}).get("columns", {})
    d["m1"]["n_rows"] = (ex.get("data_profile") or {}).get("n_rows")
    cov = {}
    for a in d["m1"]["analyzers"]:
        vals = [c["non_null"] for n, c in prof.items() if n.startswith(a + "_")]
        if vals:
            cov[a] = max(vals)
    d["m1"]["coverage"] = sorted(cov.items(), key=lambda kv: -kv[1])
    d["m1"]["results"] = read_analyzer_results(logs, d["m1"]["analyzers"])

    # --- EXPLORE ------------------------------------------------------------
    keys = ("plain_question", "observations", "takeaways", "charts", "chart_readings", "claims",
            "candidate_signals", "critique", "caveats", "recommended_confirmatory_tests",
            "adjudication", "dashboard_storyboard")
    d["explore"] = {k: ex.get(k) or ([] if k != "adjudication" else {}) for k in keys}

    # --- M2 -----------------------------------------------------------------
    a0 = by("analysis")[0]
    d["m2"] = {"conclusion": a0.get("conclusion"), "narrative": a0.get("narrative"),
               "severity": a0.get("severity"), "duration": a0.get("duration_sec")}
    stats_path = logs / "artifacts" / "c0_m2_stats_results.json"
    raw = json.loads(stats_path.read_text()) if stats_path.exists() else []
    d["m2"]["stats"] = [{k: s.get(k) for k in
                         ("tool", "config", "summary", "effect", "ci", "reject", "p_value", "underpowered")}
                        for s in raw]

    # --- M3 / M5 ------------------------------------------------------------
    dg = by("diagnosis")
    d["m3"] = {"hypotheses": dg[0]["hypotheses"] if dg else [],
               "model": dg[0].get("model_name") if dg else None,
               "duration": dg[0].get("duration_sec") if dg else None}
    sg = by("surgery")
    d["m5"] = {k: (sg[0].get(k) if sg else None) for k in
               ("module", "status", "fixed", "confidence_score", "failure_mode", "hypothesis")}
    m5p = logs / "report" / "m5_results.json"
    d["m5"]["results"] = json.loads(m5p.read_text()) if m5p.exists() else []

    # --- M4 -----------------------------------------------------------------
    f0 = by("fix")[0]
    d["m4"] = {
        "max_tier": f0["max_tier"], "fixed": f0["fixed"], "best": f0.get("best"),
        "selected_on_explore": f0.get("selected_on_explore"),
        "selection": [{k: a.get(k) for k in ("tier", "name", "verdict", "n_fixed", "n_broken", "effect")}
                      for a in (f0.get("selection_attempted") or [])],
        "confirm": {k: res.get(k) for k in
                    ("tier", "name", "kind", "source", "n_pairs", "n_baseline_correct",
                     "n_candidate_correct", "n_fixed", "n_broken", "effect", "e_value",
                     "reject", "verdict", "summary", "coverage")},
        "prompt_template": (res.get("payload") or {}).get("prompt_template", ""),
        "refine": {k: (f0.get("refine_signal") or {}).get(k) for k in ("kind", "candidate")},
        "ebh_survivors": f0.get("ebh_survivors") or [],
    }

    # --- case book: join per-case outcomes to the benchmark manifest ---------
    per_case = fixdir / "outputs.jsonl"
    cases = []
    if per_case.exists():
        for line in per_case.open():
            r = json.loads(line)
            m = manifest.get(r["case_id"])
            if m is None:
                continue  # clip not downloaded locally — nothing to show
            cases.append({"id": r["case_id"], "status": r["status"], "output": r.get("output", ""),
                          "instruction": m["instruction"], "choices": m["choices"],
                          "expected": m["expected"], "duration": m.get("duration_sec", 0.0),
                          "task": (m.get("metadata") or {}).get("mmau_task"),
                          "category": (m.get("metadata") or {}).get("category")})
    order = {"fixed": 0, "broken": 1, "unchanged": 2}
    cases.sort(key=lambda c: (order.get(c["status"], 3), c["task"] or ""))
    d["cases"] = cases
    d["_fixdir"] = str(fixdir)
    return d


def embed_figures(explore_dir: Path | None, logs: Path) -> dict:
    """basename (minus any NN_ prefix) -> data URI."""
    out = {}
    paths = []
    if explore_dir:
        paths += sorted((explore_dir / "figures").glob("*.png"))
    paths += sorted((logs / "figures").glob("*.png"))
    for p in paths:
        key = p.stem
        if len(key) > 3 and key[:2].isdigit() and key[2] == "_":
            key = key[3:]
        out[key] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return out


def embed_audio(cases: list, example_dir: Path, manifest: dict, cache: Path) -> dict:
    """Transcode each case's clip once into *cache*, then inline it."""
    if not shutil.which("ffmpeg"):
        print("! ffmpeg not on PATH — case book will render without players", file=sys.stderr)
        return {}
    cache.mkdir(parents=True, exist_ok=True)
    out, failed = {}, 0
    for c in cases:
        dst = cache / f"{c['id']}.m4a"
        if not dst.exists():
            src = example_dir / "data" / manifest[c["id"]]["audio_path"]
            if not src.exists():
                failed += 1
                continue
            p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                                *AUDIO_ARGS, str(dst)], capture_output=True)
            if p.returncode != 0:
                failed += 1
                continue
        out[c["id"]] = "data:audio/mp4;base64," + base64.b64encode(dst.read_bytes()).decode()
    if failed:
        print(f"! {failed} clip(s) could not be encoded", file=sys.stderr)
    return out


HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run-dir", default="outputs",
                help="finished run directory (default: outputs)")
ap.add_argument("--example-dir", default=str(HERE.parent),
                help="example root holding data/ (default: the parent of this script)")
ap.add_argument("--out", default=str(HERE / "index.html"), help="output html path")
ap.add_argument("--no-audio", action="store_true", help="skip clip embedding")
args = ap.parse_args()

_example_dir = Path(args.example_dir).resolve()
_logs, _explore = resolve_run(Path(args.run_dir).resolve())
_manifest = load_manifest(_example_dir)

d = collect(_logs, _explore, _manifest)
d["figures"] = embed_figures(_explore, _logs)
d["audio"] = ({} if args.no_audio
              else embed_audio(d["cases"], _example_dir, _manifest, HERE / ".audio_cache"))

print(f"run      {_logs}")
print(f"cases    {len(d['cases'])} joined to the manifest "
      f"({sum(1 for c in d['cases'] if c['status'] == 'fixed')} repaired, "
      f"{sum(1 for c in d['cases'] if c['status'] == 'broken')} broken)")
print(f"figures  {len(d['figures'])}   audio {len(d['audio'])}")

e = lambda s: html.escape(str(s), quote=True)


def pct(x, digits=1):
    return f"{x * 100:+.{digits}f}"


# ---------------------------------------------------------------- stage rail
STAGES = [
    ("pre_m1", "PRE-M1", "Probe search", "not configured", "skip"),
    ("m1", "M1", "Measure", "8 probes over 358 cases", "neutral"),
    ("m2", "M2", "Screen", "explore + 20 corrected tests", "neutral"),
    ("m3", "M3", "Explain", "1 hypothesis proposed", "neutral"),
    ("m5", "M5", "Adjudicate", "Hypothesis refuted", "warn"),
    ("m4s", "M4-SURGERY", "Intervene", "did not run", "skip"),
    ("m4", "M4-FIX", "Repair", "Validated fix, +7.6 pts", "good"),
]

rail = "\n".join(
    f'<a class="rail-item{" rail-item--skip" if tone == "skip" else ""}" href="#{sid}" data-stage="{sid}">'
    f'<span class="rail-code">{e(code)}</span>'
    f'<span class="rail-body"><span class="rail-name">{e(name)}</span>'
    f'<span class="rail-note">{e(note)}</span></span>'
    f'<span class="rail-dot rail-dot--{tone}" aria-hidden="true"></span></a>'
    for sid, code, name, note, tone in STAGES
)

# ---------------------------------------------------------------- stat tiles
cfm = d['m4']['confirm']
base_acc = cfm['n_baseline_correct'] / cfm['n_pairs']
cand_acc = cfm['n_candidate_correct'] / cfm['n_pairs']

TILES = [
    ("Cases evaluated", "896", "358 diagnose &middot; 538 confirm", ""),
    ("Baseline accuracy", f"{base_acc:.1%}", f"{cfm['n_baseline_correct']}/{cfm['n_pairs']} on the confirm split", ""),
    ("Hypothesis verdict", "Refuted", "M5 rejected M3&rsquo;s explanation", "warn"),
    ("Repair effect", f"{pct(cfm['effect'], 2)} pts", f"{cfm['n_fixed']} repaired &middot; {cfm['n_broken']} broken", "good"),
    ("Evidence strength", "e = 6,281", "McNemar paired, H&#8320; rejected", "good"),
]
tiles = "\n".join(
    f'<div class="tile{" tile--" + tone if tone else ""}">'
    f'<div class="tile-label">{lab}</div><div class="tile-value">{val}</div>'
    f'<div class="tile-note">{note}</div></div>'
    for lab, val, note, tone in TILES
)

# ---------------------------------------------------------------- M1
n_rows_total = d['m1']['n_rows'] or 358
cov_max = max((r['n'] or 0) for r in d['m1']['results']) or 1
cov_rows = []
for r in d['m1']['results']:
    n = r['n'] or 0
    hl = "".join(
        f'<div class="hl"><span class="hl-v">{h["value"]}</span>'
        f'<span class="hl-l">{e(h["label"])}</span></div>'
        if h["value"] is not None else
        f'<div class="hl hl--none"><span class="hl-v">&mdash;</span>'
        f'<span class="hl-l">{e(h["label"])}: not scoreable here</span></div>'
        for h in r['headline']) or '<div class="hl hl--none"><span class="hl-v">&mdash;</span>' \
                                   '<span class="hl-l">produced no case-level numbers</span></div>'
    if r['n'] is None:
        count_cell = '<td class="num muted">run<br>level</td>'
        bar_cell = ('<td class="barcell"><span class="barpct">aggregate only &mdash;'
                    '<br>no per-case column</span></td>')
    else:
        count_cell = f'<td class="num">{n}</td>'
        bar_cell = (f'<td class="barcell"><span class="bar" style="--w:{n / cov_max:.4f}"></span>'
                    f'<span class="barpct">{n / n_rows_total:.0%} of cases</span></td>')
    cov_rows.append(
        f'<tr><td><div class="an">{e(r["name"])}</div>'
        f'<div class="anq">{e(r["question"])}</div></td>'
        f'{count_cell}{bar_cell}'
        f'<td class="hlcell">{hl}</td></tr>')
cov_rows = "\n".join(cov_rows)
silent = [r['name'] for r in d['m1']['results']
          if r['name'] not in dict(d['m1']['coverage'])]

# ---------------------------------------------------------------- EXPLORE
figs = d['figures']


def figure_block(name, caption, reading=None, dni=None):
    if name not in figs:
        return ""
    parts = [f'<figure class="fig"><img src="{figs[name]}" alt="{e(caption)}" loading="lazy">',
             f'<figcaption><span class="fig-cap">{e(caption)}</span>']
    if reading:
        parts.append(f'<span class="fig-read">{e(reading)}</span>')
    if dni:
        parts.append(f'<span class="fig-dni"><b>Does not show</b> {e(dni)}</span>')
    parts.append('</figcaption></figure>')
    return "".join(parts)


readings = {r['chart']: r for r in d['explore']['chart_readings']}
chart_display = {c['name']: (c.get('display_name') or c.get('title') or c['name'])
                 for c in d['explore']['charts']}

#: The four that carry the M2 story: the ranking itself, then the three signals
#: it ranks highest. Everything else the exploration rendered is provenance and
#: lives behind a disclosure.
LEAD_CHARTS = ["top_discriminators", "confidence_logprob_by_outcome",
               "format_flip_rate_by_outcome", "positional_bias_by_outcome"]

#: Charts the explorer rendered outside its declared `charts` list carry no
#: display name; fall back to the title drawn inside the figure itself.
EXTRA_TITLES = {
    "confidence_logprob_by_outcome": "Model confidence (log-prob) by outcome",
    "format_flip_rate_by_outcome": "Answer flip rate under format changes",
    "positional_bias_by_outcome": "Positional answer bias by outcome",
    "scatter_flip_vs_bias": "Answer flips vs positional bias",
    "corr_heatmap": "Correlation between recorded signals",
}


def _fig(name):
    r = readings.get(name, {})
    caption = (chart_display.get(name) or EXTRA_TITLES.get(name)
               or name.replace('_', ' ').capitalize())
    return figure_block(name, caption, r.get('reading'), r.get('do_not_infer'))


charts_lead = "\n".join(b for b in (_fig(n) for n in LEAD_CHARTS) if b)
_shown = set(LEAD_CHARTS) | {"class_balance", "m2_effects"}
charts_rest = "\n".join(
    b for b in (_fig(n) for n in sorted(figs) if n not in _shown) if b)

takeaways = "\n".join(
    f'<details class="take"{" open" if i < 2 else ""}>'
    f'<summary><span class="take-plain">{e(t["plain_title"])}</span>'
    f'<span class="take-title">{e(t["title"])}</span></summary>'
    f'<p>{e(t["analysis"])}</p>'
    + (f'<p class="caveat"><b>Caveat</b> {e(t["caveat"])}</p>' if t.get('caveat') else '')
    + '</details>'
    for i, t in enumerate(d['explore']['takeaways'])
)

adj = d['explore']['adjudication']
signals = "\n".join(
    f'<li><span class="sig-name">{e(s["display_name"])}</span>'
    f'<span class="sig-why">{e(s["rationale"])}</span>'
    f'<span class="sig-test">{e(s["suggested_test"])}</span></li>'
    for s in d['explore']['candidate_signals']
)

caveats = "\n".join(f'<li>{e(c)}</li>' for c in d['explore']['caveats'])
critique = "\n".join(f'<li>{e(c)}</li>' for c in d['explore']['critique'])

# ---------------------------------------------------------------- M2
stats = sorted(d['m2']['stats'], key=lambda s: (not s['reject'], -abs(s.get('effect') or 0)))
stat_rows = []
for s in stats:
    sig = (s.get('config') or {}).get('signal') or json.dumps(s.get('config') or {})
    ci = s.get('ci') or [None, None]
    ci_s = f"{ci[0]:+.3f} .. {ci[1]:+.3f}" if ci[0] is not None else "&mdash;"
    p = s.get('p_value')
    p_s = f"{p:.2e}" if isinstance(p, (int, float)) else "&mdash;"
    verdict = ('<span class="chip chip--good">rejected H&#8320;</span>' if s['reject']
               else '<span class="chip chip--mute">inconclusive</span>')
    stat_rows.append(
        f'<tr><td class="mono">{e(sig)}</td><td class="mono muted">{e(s["tool"])}</td>'
        f'<td class="num">{(s.get("effect") or 0):+.4f}</td>'
        f'<td class="num muted mono">{ci_s}</td><td class="num muted mono">{p_s}</td>'
        f'<td>{verdict}</td></tr>')
n_reject = sum(1 for x in stats if x['reject'])
stat_rows_sig = "\n".join(stat_rows[:n_reject])
stat_rows_null = "\n".join(stat_rows[n_reject:])
n_null = len(stats) - n_reject

# ---------------------------------------------------------------- M3 / M5
hyp = d['m3']['hypotheses'][0]
m5r = d['m5']['results'][0]
fdr = m5r['evidence'].get('fdr') or {}

# ---------------------------------------------------------------- M4
sel = d['m4']['selection']
maxabs = max(abs(s['effect']) for s in sel) * 1.15
VERDICT_TONE = {'fixed': 'good', 'partial': 'warn', 'regressed': 'bad', 'unsafe': 'bad'}
sel_rows = []
for s in sel:
    w = abs(s['effect']) / maxabs * 50
    side = 'pos' if s['effect'] >= 0 else 'neg'
    tone = VERDICT_TONE.get(s['verdict'], 'mute')
    is_win = s['name'] == d['m4']['selected_on_explore']
    winner = 'row--winner' if is_win else ''
    star = '<span class="star">selected</span>' if is_win else ''
    sel_rows.append(
        f'<tr class="{winner}"><td class="mono tier">{e(s["tier"])}</td>'
        f'<td class="mono">{e(s["name"])} {star}</td>'
        f'<td><span class="chip chip--{tone}">{e(s["verdict"])}</span></td>'
        f'<td class="num">{s["n_fixed"]}</td><td class="num">{s["n_broken"]}</td>'
        f'<td class="num eff eff--{side}">{pct(s["effect"], 2)}</td>'
        f'<td class="divcell"><span class="divaxis"></span>'
        f'<span class="divbar divbar--{side}" style="--w:{w:.3f}%"></span></td></tr>')
sel_rows = "\n".join(sel_rows)

cases = d['cases']
audio = d['audio']
n_fixed = sum(1 for c in cases if c['status'] == 'fixed')
n_broken = sum(1 for c in cases if c['status'] == 'broken')

# ---------------------------------------------------------------- assemble
PAGE = r"""<title>Qwen2-Audio on MMAU</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#fbfbfa; --surface:#ffffff; --surface-2:#f4f5f6; --line:#e0e3e5; --line-soft:#eceef0;
  --ink:#15191c; --ink-2:#3d464d; --ink-3:#5c6670; --ink-4:#8b949b;
  --accent:#0a90a3; --accent-soft:#e3f2f4;
  --good:#0f8055; --good-soft:#e2f1ea; --bad:#c0392b; --bad-soft:#fbe9e6;
  --warn-ink:#8a6410; --warn:#e0a11a; --warn-soft:#fbf0d9;
  --shadow:0 1px 2px rgba(20,28,33,.05),0 4px 14px rgba(20,28,33,.05);
  --r:7px;
  --f-display:"IBM Plex Sans Condensed",ui-sans-serif,system-ui,sans-serif;
  --f-body:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#121517; --surface:#181c1f; --surface-2:#1f2429; --line:#2b3136; --line-soft:#23282c;
    --ink:#e7ebed; --ink-2:#bcc5cb; --ink-3:#8e9aa3; --ink-4:#6b767e;
    --accent:#2a9fb2; --accent-soft:#123037;
    --good:#2aa578; --good-soft:#102b22; --bad:#e2604c; --bad-soft:#31201d;
    --warn-ink:#d9a441; --warn:#d9a441; --warn-soft:#312716;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
  }
}
:root[data-theme="dark"]{
  --ground:#121517; --surface:#181c1f; --surface-2:#1f2429; --line:#2b3136; --line-soft:#23282c;
  --ink:#e7ebed; --ink-2:#bcc5cb; --ink-3:#8e9aa3; --ink-4:#6b767e;
  --accent:#2a9fb2; --accent-soft:#123037;
  --good:#2aa578; --good-soft:#102b22; --bad:#e2604c; --bad-soft:#31201d;
  --warn-ink:#d9a441; --warn:#d9a441; --warn-soft:#312716;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 4px 14px rgba(0,0,0,.25);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--f-body); font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
h1,h2,h3{font-family:var(--f-display); font-weight:700; text-wrap:balance; margin:0; line-height:1.2}
p{margin:0}
a{color:var(--accent)}
.mono{font-family:var(--f-mono); font-size:.86em}
.num{font-variant-numeric:tabular-nums; text-align:right; font-family:var(--f-mono); font-size:.86em}
.muted{color:var(--ink-3)}
:where(a,button,summary,input,select):focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:3px}

/* ---------- shell ---------- */
.wrap{max-width:1180px; margin:0 auto; padding:0 24px}
.shell{display:grid; grid-template-columns:232px minmax(0,1fr); gap:44px; align-items:start}

/* ---------- header ---------- */
header.top{border-bottom:1px solid var(--line); background:var(--surface)}
.top-in{padding:38px 0 30px; display:flex; flex-direction:column; gap:14px}
.eyebrow{font-family:var(--f-mono); font-size:11.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--accent); font-weight:500}
h1{font-size:clamp(30px,4.4vw,46px); letter-spacing:-.015em}
.sub{max-width:66ch; color:var(--ink-2); font-size:16px}
.meta{display:flex; flex-wrap:wrap; gap:7px; margin-top:4px}
.meta span{font-family:var(--f-mono); font-size:11.5px; color:var(--ink-3); background:var(--surface-2);
  border:1px solid var(--line-soft); border-radius:99px; padding:3px 10px}

/* ---------- headline band ---------- */
.band{background:var(--surface-2); border-bottom:1px solid var(--line)}
.band-in{padding:26px 0 30px; display:flex; flex-direction:column; gap:20px}
.arc{display:flex; gap:14px; align-items:flex-start; max-width:78ch}
.arc-mark{flex:none; width:3px; align-self:stretch; background:var(--accent); border-radius:2px}
.arc p{font-size:16.5px; color:var(--ink)}
.arc b{font-weight:600}
.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:var(--r); overflow:hidden}
.tile{background:var(--surface); padding:15px 17px 17px; display:flex; flex-direction:column; gap:3px}
.tile-label{font-family:var(--f-mono); font-size:10.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink-3)}
.tile-value{font-family:var(--f-display); font-size:27px; font-weight:700; letter-spacing:-.01em;
  font-variant-numeric:tabular-nums; line-height:1.15}
.tile-note{font-size:12.5px; color:var(--ink-3); line-height:1.45}
.tile--good .tile-value{color:var(--good)}
.tile--warn .tile-value{color:var(--warn-ink)}

/* ---------- rail ---------- */
nav.rail{position:sticky; top:20px; padding:34px 0; display:flex; flex-direction:column; gap:2px}
.rail-head{font-family:var(--f-mono); font-size:10.5px; letter-spacing:.11em; text-transform:uppercase;
  color:var(--ink-4); padding:0 10px 10px}
.rail-item{display:grid; grid-template-columns:54px minmax(0,1fr) 8px; align-items:center; gap:10px;
  padding:9px 10px; border-radius:var(--r); text-decoration:none; color:inherit; position:relative;
  border:1px solid transparent}
.rail-item:hover{background:var(--surface-2)}
.rail-item.is-active{background:var(--surface); border-color:var(--line); box-shadow:var(--shadow)}
.rail-code{font-family:var(--f-mono); font-size:10px; font-weight:600; color:var(--accent);
  letter-spacing:.04em; white-space:nowrap}
.rail-body{display:flex; flex-direction:column; min-width:0}
.rail-name{font-family:var(--f-display); font-weight:600; font-size:14.5px}
.rail-note{font-size:11.5px; color:var(--ink-3); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.rail-dot{width:8px; height:8px; border-radius:50%; background:var(--ink-4)}
.rail-dot--good{background:var(--good)} .rail-dot--warn{background:var(--warn)}
.rail-dot--skip{background:transparent; border:1px dashed var(--ink-4)}
.rail-item--skip .rail-code,.rail-item--skip .rail-name{color:var(--ink-4)}
.rail-item--skip .rail-name{font-weight:500}

/* ---------- stages ---------- */
main{padding:34px 0 80px; display:flex; flex-direction:column; gap:14px; min-width:0}
.stage{scroll-margin-top:24px; background:var(--surface); border:1px solid var(--line);
  border-radius:var(--r); box-shadow:var(--shadow); overflow:hidden}
.stage-head{display:flex; flex-wrap:wrap; gap:10px 16px; align-items:baseline;
  padding:20px 24px; border-bottom:1px solid var(--line-soft); background:var(--surface)}
.stage-code{font-family:var(--f-mono); font-size:11.5px; font-weight:600; letter-spacing:.11em;
  color:var(--accent); background:var(--accent-soft); padding:4px 9px; border-radius:4px}
.stage-title{font-size:22px; letter-spacing:-.01em}
.stage-sub{flex:1 1 100%; color:var(--ink-3); font-size:13.5px; margin-top:-2px}
.stage--skip{border-style:dashed; box-shadow:none; background:transparent}
.stage--skip .stage-head{background:transparent}
.stage--skip .stage-title{color:var(--ink-3); font-weight:600}
.stage-code--skip{color:var(--ink-3); background:var(--surface-2)}
.stage-time{font-family:var(--f-mono); font-size:11.5px; color:var(--ink-4); margin-left:auto}
.stage-body{padding:22px 24px 26px; display:flex; flex-direction:column; gap:20px}
.stage-body > p{max-width:68ch; color:var(--ink-2)}
.lede{font-family:var(--f-display); font-size:22px; line-height:1.4; font-weight:600;
  color:var(--ink); max-width:36ch; letter-spacing:-.005em; text-wrap:balance}
.lede--warn{color:var(--warn-ink)} .lede--good{color:var(--good)} .lede--skip{color:var(--ink-3)}
.ledewrap{display:grid; grid-template-columns:minmax(0,1fr); gap:12px;
  padding-bottom:4px; border-bottom:1px solid var(--line-soft)}
@media (min-width:820px){.ledewrap{grid-template-columns:minmax(0,1.05fr) minmax(0,1fr); gap:32px}}
.ledewrap > p{color:var(--ink-2); font-size:14px; max-width:56ch; margin:0}
.key{font-family:var(--f-mono); font-size:.94em; font-weight:600; color:var(--accent);
  background:var(--accent-soft); padding:1px 5px; border-radius:3px; white-space:nowrap}
.key--good{color:var(--good); background:var(--good-soft)}
.key--bad{color:var(--bad); background:var(--bad-soft)}
.key--warn{color:var(--warn-ink); background:var(--warn-soft)}
h3{font-size:13px; font-family:var(--f-mono); font-weight:600; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink-3)}
.block{display:flex; flex-direction:column; gap:11px}

/* ---------- tables ---------- */
.scroll{overflow-x:auto; border:1px solid var(--line-soft); border-radius:var(--r)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
th{font-family:var(--f-mono); font-size:10.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--ink-3); font-weight:500; text-align:left; padding:9px 13px; background:var(--surface-2);
  border-bottom:1px solid var(--line); white-space:nowrap}
th.num{text-align:right}
td{padding:8px 13px; border-bottom:1px solid var(--line-soft); vertical-align:middle}
tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface-2)}
.row--winner{background:var(--good-soft)}
.row--winner:hover{background:var(--good-soft)}
.star{font-family:var(--f-body); font-size:10.5px; font-weight:600; color:var(--good);
  border:1px solid var(--good); border-radius:99px; padding:1px 7px; margin-left:7px; white-space:nowrap}
.tier{color:var(--ink-3)}

/* ---------- bars ---------- */
.barcell{width:170px; padding-right:18px}
.bar{display:block; height:9px; border-radius:0 4px 4px 0; background:var(--accent);
  width:calc(var(--w) * 100%); min-width:3px}
.barpct{display:block; margin-top:4px; font-family:var(--f-mono); font-size:11px; color:var(--ink-3)}
.an{font-family:var(--f-mono); font-size:12.5px; color:var(--ink)}
.anq{font-size:12.5px; color:var(--ink-3); line-height:1.45; max-width:38ch; margin-top:2px}
.hlcell{min-width:230px}
.hl{display:flex; gap:8px; align-items:baseline; padding:2px 0}
.hl-v{font-family:var(--f-mono); font-size:14px; font-variant-numeric:tabular-nums;
  color:var(--ink); min-width:56px; text-align:right}
.hl-l{font-size:12px; color:var(--ink-3); line-height:1.4}
.hl--none .hl-v{color:var(--ink-4)}
.eff{white-space:nowrap; padding-right:14px}
.eff--pos{color:var(--good)} .eff--neg{color:var(--bad)}
.divcell{position:relative; width:210px; min-width:210px; height:30px; padding:0}
.divaxis{position:absolute; left:50%; top:4px; bottom:4px; width:1px; background:var(--line)}
.divbar{position:absolute; top:11px; height:9px; width:var(--w); min-width:2px}
.divbar--pos{left:50%; background:var(--good); border-radius:0 4px 4px 0}
.divbar--neg{right:50%; background:var(--bad); border-radius:4px 0 0 4px}

/* ---------- chips ---------- */
.chip{display:inline-block; font-family:var(--f-mono); font-size:10.5px; font-weight:500;
  padding:2px 8px; border-radius:99px; white-space:nowrap; border:1px solid transparent}
.chip--good{color:var(--good); background:var(--good-soft); border-color:var(--good)}
.chip--bad{color:var(--bad); background:var(--bad-soft); border-color:var(--bad)}
.chip--warn{color:var(--warn-ink); background:var(--warn-soft); border-color:var(--warn)}
.chip--mute{color:var(--ink-3); background:var(--surface-2); border-color:var(--line)}

/* ---------- figures ---------- */
.figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:16px}
.fig{margin:0; border:1px solid var(--line-soft); border-radius:var(--r); overflow:hidden; background:var(--surface)}
.fig img{display:block; width:100%; height:auto; background:#fff}
figcaption{padding:11px 13px 13px; display:flex; flex-direction:column; gap:5px; border-top:1px solid var(--line-soft)}
.fig-cap{font-family:var(--f-display); font-weight:600; font-size:14px}
.fig-read{font-size:12.5px; color:var(--ink-2); line-height:1.5}
.fig-dni{font-size:11.5px; color:var(--ink-3); line-height:1.45}
.fig-dni b{font-family:var(--f-mono); font-size:10px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-4)}

/* ---------- takeaways ---------- */
.take{border:1px solid var(--line-soft); border-radius:var(--r); background:var(--surface); overflow:hidden}
.take + .take{margin-top:8px}
.take summary{padding:12px 15px; cursor:pointer; list-style:none;
  display:grid; grid-template-columns:minmax(0,1fr) 16px; grid-auto-rows:min-content; gap:3px 12px}
.take summary::-webkit-details-marker{display:none}
.take summary::after{content:"+"; grid-column:2; grid-row:1/span 2; align-self:start;
  font-family:var(--f-mono); font-size:14px; line-height:1.35; color:var(--ink-4); text-align:right}
.take[open] summary::after{content:"\2212"}
.take summary > *{grid-column:1}
.take summary:hover{background:var(--surface-2)}
.take-plain{font-family:var(--f-display); font-weight:600; font-size:15px}
.take-title{font-size:12.5px; color:var(--ink-3); line-height:1.45}
.take p{padding:0 15px 14px; font-size:13.5px; color:var(--ink-2); max-width:74ch}
.take .caveat{color:var(--ink-3); font-size:12.5px}
.take .caveat b{font-family:var(--f-mono); font-size:10px; letter-spacing:.06em; text-transform:uppercase}

/* ---------- callouts / lists ---------- */
.note{border-left:3px solid var(--accent); background:var(--surface-2); padding:13px 16px;
  border-radius:0 var(--r) var(--r) 0; font-size:13.5px; color:var(--ink-2); max-width:74ch}
.note--warn{border-left-color:var(--warn)}
.note--good{border-left-color:var(--good)}
.note b{color:var(--ink); font-weight:600}
ul.plain{margin:0; padding-left:19px; display:flex; flex-direction:column; gap:6px;
  font-size:13px; color:var(--ink-2); max-width:76ch}
ul.sigs{margin:0; padding:0; list-style:none; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(268px,1fr)); gap:11px}
ul.sigs li{border:1px solid var(--line-soft); border-radius:var(--r); padding:12px 14px;
  display:flex; flex-direction:column; gap:5px; background:var(--surface)}
.sig-name{font-family:var(--f-display); font-weight:600; font-size:14px}
.sig-why{font-size:12.5px; color:var(--ink-2); line-height:1.5}
.sig-test{font-size:11.5px; color:var(--ink-3); font-family:var(--f-mono); line-height:1.45}
.hyp{border:1px solid var(--line); border-radius:var(--r); padding:17px 19px; background:var(--surface-2);
  display:flex; flex-direction:column; gap:9px}
.hyp-state{font-size:16px; line-height:1.55; max-width:72ch}
.hyp-meta{display:flex; flex-wrap:wrap; gap:8px; align-items:center}
pre.tmpl{margin:0; background:var(--surface-2); border:1px solid var(--line-soft); border-radius:var(--r);
  padding:14px 16px; overflow-x:auto; font-family:var(--f-mono); font-size:12.5px; line-height:1.65;
  color:var(--ink-2); white-space:pre-wrap; word-break:break-word}
.kv{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:var(--r); overflow:hidden}
@media (max-width:720px){.kv{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:430px){.kv{grid-template-columns:1fr}}
.kv > div{background:var(--surface); padding:11px 14px; display:flex; flex-direction:column; gap:2px}
.kv dt{font-family:var(--f-mono); font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3)}
.kv dd{margin:0; font-family:var(--f-mono); font-size:14px; font-variant-numeric:tabular-nums;
  color:var(--ink); overflow-wrap:anywhere; line-height:1.4}

/* ---------- case book ---------- */
.filters{display:flex; flex-wrap:wrap; gap:8px; align-items:center}
.filters button{font-family:var(--f-mono); font-size:11.5px; padding:5px 12px; border-radius:99px;
  border:1px solid var(--line); background:var(--surface); color:var(--ink-2); cursor:pointer}
.filters button:hover{border-color:var(--ink-4)}
.filters button[aria-pressed="true"]{background:var(--ink); color:var(--ground); border-color:var(--ink)}
.filters .count{font-family:var(--f-mono); font-size:11.5px; color:var(--ink-3); margin-left:auto}
.cases{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:13px}
.case{border:1px solid var(--line); border-radius:var(--r); background:var(--surface); overflow:hidden;
  display:flex; flex-direction:column}
.case-top{display:flex; gap:8px; align-items:center; padding:11px 14px; border-bottom:1px solid var(--line-soft);
  background:var(--surface-2); flex-wrap:wrap}
.case-task{font-family:var(--f-mono); font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--ink-3)}
.case-dur{font-family:var(--f-mono); font-size:10.5px; color:var(--ink-4); margin-left:auto}
.case-q{padding:13px 14px 0; font-family:var(--f-display); font-weight:600; font-size:14.5px; line-height:1.35}
.case audio{width:calc(100% - 28px); margin:12px 14px 0; height:34px}
.opts{list-style:none; margin:12px 0 0; padding:0 14px; display:flex; flex-direction:column; gap:4px}
.opts li{font-size:12.5px; color:var(--ink-2); line-height:1.4; padding:4px 8px; border-radius:4px;
  border:1px solid transparent; display:flex; gap:7px}
.opts li .lt{font-family:var(--f-mono); font-weight:600; color:var(--ink-3); flex:none}
.opts li.is-expected{background:var(--good-soft); border-color:var(--good); color:var(--ink)}
.opts li.is-expected .lt{color:var(--good)}
.case-foot{margin-top:auto; padding:12px 14px; display:flex; gap:8px; align-items:center;
  border-top:1px solid var(--line-soft); flex-wrap:wrap}
.case-foot .lbl{font-family:var(--f-mono); font-size:10.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--ink-4)}
.case-foot .ans{font-family:var(--f-mono); font-size:12.5px; color:var(--ink)}
.case-id{font-family:var(--f-mono); font-size:10px; color:var(--ink-4); margin-left:auto}

footer{border-top:1px solid var(--line); background:var(--surface); padding:26px 0 40px;
  font-size:12.5px; color:var(--ink-3)}
footer .wrap{display:flex; flex-direction:column; gap:6px}
footer code{font-family:var(--f-mono); font-size:11.5px; color:var(--ink-2)}

@media (max-width:900px){
  .shell{grid-template-columns:1fr; gap:0}
  nav.rail{position:static; flex-direction:row; overflow-x:auto; padding:14px 0; gap:8px;
    border-bottom:1px solid var(--line)}
  .rail-head{display:none}
  .rail-item{flex:none; grid-template-columns:auto auto; border:1px solid var(--line); background:var(--surface)}
  .rail-note,.rail-dot{display:none}
  .rail-item .rail-body{flex-direction:row; gap:6px}
  .divcell{width:170px}
  .barcell{width:120px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important; scroll-behavior:auto!important}}
html{scroll-behavior:smooth}
</style>

<header class="top"><div class="wrap"><div class="top-in">
  <div class="eyebrow">EvalVitals &middot; diagnostic loop &middot; full M1&ndash;M5 trace</div>
  <h1>Qwen2-Audio on MMAU</h1>
  <p class="sub">One complete pass of the diagnosis loop &mdash; measure, screen, explain, adjudicate,
    repair &mdash; over four-way multiple-choice questions about short sound, music and speech clips.</p>
  <div class="meta">__META__</div>
</div></div></header>

<section class="band"><div class="wrap"><div class="band-in">
  <div class="arc"><span class="arc-mark"></span><p>
    The loop proposed one explanation for the model&rsquo;s failures and <b>the held-out split refuted
    it</b> &mdash; yet the repair search still found an intervention worth <b>+7.62 points</b> on unseen
    cases. <b>A wrong diagnosis and a working fix are separate questions</b>, and this run answers them
    separately.
  </p></div>
  <div class="tiles">__TILES__</div>
</div></div></section>

<div class="wrap"><div class="shell">
<nav class="rail" aria-label="Pipeline stages">
  <div class="rail-head">Pipeline</div>
  __RAIL__
</nav>

<main>

<!-- ============ pre-M1 ============ -->
<section class="stage stage--skip" id="pre_m1">
  <div class="stage-head"><span class="stage-code stage-code--skip">PRE-M1</span>
    <h2 class="stage-title">Probe search</h2>
    <span class="stage-time">not configured</span>
    <p class="stage-sub">Optional. Synthesizes <em>new</em> failing cases instead of analysing existing ones &mdash; its output is data, not a verdict.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede lede--skip">Did not run &mdash; the case set is fixed, not grown.</p>
      <p>Every case below came from the MMAU test-mini manifest as published. Shown because a stage that
        did not run is information: no synthetic failures entered this run.</p>
    </div>
  </div>
</section>

<!-- ============ M1 ============ -->
<section class="stage" id="m1">
  <div class="stage-head"><span class="stage-code">M1</span><h2 class="stage-title">Measure</h2>
    <span class="stage-time">253 s</span>
    <p class="stage-sub">Eight probes, each measuring as much of the split as it can afford. No probe judges anything &mdash; they only produce numbers.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede">The model is confidently and repeatably wrong.</p>
      <p>It does not hesitate (<span class="key">entropy 0.10</span>), does not waver
        (<span class="key">5 of 5 samples identical</span>), and stays confident when wrong
        (<span class="key key--bad">calibration error 0.34</span>). On half the cases probed
        <span class="key key--bad">none of 5 attempts</span> contains the right answer &mdash; that is
        not knowing, not bad luck. And <span class="key key--bad">29&#37; of answers</span> change if you
        only reorder the options.</p>
    </div>

    <div class="block"><h3>The eight probes and what each one measured</h3>
      <div class="scroll"><table>
        <thead><tr><th>Probe &mdash; and the question it asks</th><th class="num">Cases<br>measured</th>
          <th>Coverage of the 358&#8209;case split</th><th>What it measured</th></tr></thead>
        <tbody>__COV__</tbody></table></div>
    </div>

    <div class="note note--warn"><b>Coverage is the caveat that travels with every number below.</b>
      The expensive probes run on subsamples, so coverage ranges from
      <span class="key key--warn">200 of 358</span> cases down to <span class="key key--warn">16</span>.
      A finding measured on 16 cases and one measured on 200 are not the same kind of fact, and nothing
      downstream re-levels them.</div>

    <details class="take"><summary><span class="take-plain">Two probes came back without a usable signal</span>
      <span class="take-title">__SILENT__ reported run-level aggregates only; selfcheck_consistency covered 32 cases and still measured nothing.</span></summary>
      <p>__SILENT__ produced real numbers, but no per-case column &mdash; so no later chart or test can
        use them; they shape the written analysis without ever appearing in a figure.
        <span class="mono">selfcheck_consistency</span> is the other kind of gap: its metric needs a
        sentence to check, and the model answers with a bare letter. Neither absence is evidence of
        health &mdash; in both cases the probe simply could not see anything.</p></details>

    <details class="take"><summary><span class="take-plain">The task, verbatim from the run</span>
      <span class="take-title">The ExperimentProtocol the loop was given, unedited.</span></summary>
      <p>__PROTO__</p></details>

    <div class="figs">__FIG_BALANCE__</div>
  </div>
</section>

<!-- ============ M2 (explore + screen) ============ -->
<section class="stage" id="m2">
  <div class="stage-head"><span class="stage-code">M2</span><h2 class="stage-title">Screen</h2>
    <span class="stage-time">545 s</span>
    <p class="stage-sub">Join M1&rsquo;s numbers to the PASS/FAIL labels, find which ones track failure, correct for multiple testing.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede">The failures are a real audio-grounding gap, not a scoring artifact.</p>
      <p>Wrong answers are not mis-parsed and not truncated away &mdash; when the model is wrong it is
        wrong on every resample. Two amplifiers sit on top: <span class="key key--bad">overconfidence</span>
        and a <span class="key key--bad">letter-position prior</span> it falls back on when the audio does
        not settle the question. <span class="key">__NREJ__ of __NSTAT__</span> tests survived
        Benjamini&ndash;Hochberg correction.</p>
    </div>

    <div class="block"><h3>Which signals separate right answers from wrong ones</h3>
      <div class="figs">__FIG_TOP__</div>
      <div class="scroll"><table>
        <thead><tr><th>Signal that tracks failure</th><th>Test</th><th class="num">Effect</th>
          <th class="num">95&#37; CI</th><th class="num">p</th><th>Verdict</th></tr></thead>
        <tbody>__STATS_SIG__</tbody></table></div>
      <p class="muted" style="font-size:13px">Effect is the standardized FAIL-vs-PASS separation;
        positive means the signal runs higher on failures.</p>
      <details class="take"><summary><span class="take-plain">__NNULL__ further tests came back inconclusive</span>
        <span class="take-title">Tested and not significant &mdash; recorded so the family size is visible, since it is what the correction was applied over.</span></summary>
        <div style="padding:0 15px 15px"><div class="scroll"><table>
          <thead><tr><th>Signal</th><th>Test</th><th class="num">Effect</th><th class="num">95&#37; CI</th>
            <th class="num">p</th><th>Verdict</th></tr></thead>
          <tbody>__STATS_NULL__</tbody></table></div></div></details>
    </div>

    <div class="block"><h3>What M2 handed to M3</h3>
      <div class="note note--good">__CONCLUSION__</div></div>

    <div class="note note--warn"><b>Half of this stage is exploratory, and does not carry the same weight.</b>
      An agent wrote its own analysis code over M1&rsquo;s table and proposed
      <span class="key key--warn">__NCAND__ candidate signals</span>; e-BH rejected the null for all
      __NADJ__ &mdash; but on the same rows that discovered them, with thresholds frozen from those rows.
      Only the tests in the table above are corrected and held-out grade. Coverage compounds it: several
      signals are scored on 16&ndash;48 cases.</div>

    <details class="take"><summary><span class="take-plain">The other nine figures</span>
      <span class="take-title">Everything else the exploration rendered, kept for provenance.</span></summary>
      <div style="padding:0 15px 15px"><div class="figs">__FIG_REST__</div></div></details>

    <details class="take"><summary><span class="take-plain">The exploration&rsquo;s own findings and caveats</span>
      <span class="take-title">Eight plain-language takeaways, plus the limits and method choices the agent flagged itself.</span></summary>
      <div style="padding:0 15px 15px; display:flex; flex-direction:column; gap:10px">
        __TAKE__
        <h3>Limits it flagged</h3><ul class="plain">__CAVEATS__</ul>
        <h3>Method notes</h3><ul class="plain">__CRITIQUE__</ul>
      </div></details>
  </div>
</section>

<!-- ============ M3 ============ -->
<section class="stage" id="m3">
  <div class="stage-head"><span class="stage-code">M3</span><h2 class="stage-title">Explain</h2>
    <span class="stage-time">301 s</span>
    <p class="stage-sub">A judge model reads the analysis and proposes a falsifiable root cause.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede">One hypothesis: the model&rsquo;s stated confidence is a reflex, not introspection.</p>
      <p>A sharp claim, and that is the point &mdash; it names a measurable quantity, predicts a
        direction, and can therefore be checked on data the diagnosis never saw. M5 does exactly that
        next, and it does not survive.</p>
    </div>
    <div class="hyp">
      <div class="hyp-meta"><span class="chip chip--mute">failure mode</span>
        <span class="mono">__FM__</span></div>
      <p class="hyp-state">__HYP__</p>
    </div>
    <p>Exactly one hypothesis came out of this cycle. It is a sharp claim &mdash; it names a measurable
      quantity (verbalized confidence), predicts a direction (worse discrimination than log-prob
      confidence), and so can be checked on data the diagnosis never saw. That is what M5 does next.</p>
  </div>
</section>

<!-- ============ M5 ============ -->
<section class="stage" id="m5">
  <div class="stage-head"><span class="stage-code">M5</span><h2 class="stage-title">Adjudicate</h2>
    <span class="stage-time">4 s</span>
    <p class="stage-sub">Re-run the probes on a held-out split and test the hypothesis as stated.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede lede--warn">Refuted &mdash; the test came back significant in the opposite direction.</p>
      <p>The hypothesis predicted that stated confidence would discriminate <em>worse</em> than log-prob
        confidence. Log-prob confidence separated PASS from FAIL strongly and against the predicted sign
        (<span class="key key--warn">effect &minus;0.41</span>), so the claim as written does not hold.</p>
    </div>
    <div class="note note--warn"><b>Refuted.</b> The test the hypothesis asked for came back significant
      &mdash; in the opposite direction. Verbalized confidence did not discriminate <em>less</em> than
      log-prob confidence; the log-prob channel separated PASS from FAIL strongly and against the
      predicted sign, so the claim as written does not hold.</div>
    <div class="kv">
      <div><dt>Status</dt><dd style="color:var(--warn-ink)">refuted</dd></div>
      <div><dt>Effect size</dt><dd>__M5EFF__</dd></div>
      <div><dt>95&#37; CI</dt><dd>__M5CI__</dd></div>
      <div><dt>Confidence</dt><dd>__M5CONF__</dd></div>
      <div><dt>BH-corrected</dt><dd>survived</dd></div>
      <div><dt>Evidence grade</dt><dd>__M5GRADE__</dd></div>
    </div>
    <div class="block"><h3>Verdict string, verbatim</h3>
      <pre class="tmpl">__M5VERDICT__</pre></div>
    <p class="muted" style="font-size:13px">Multiplicity was handled with Benjamini&ndash;Hochberg across
      __NTESTED__ tests at &alpha;=0.05; this signal was among those that survived correction.</p>
  </div>
</section>

<!-- ============ M4-surgery ============ -->
<section class="stage stage--skip" id="m4s">
  <div class="stage-head"><span class="stage-code stage-code--skip">M4-SURGERY</span>
    <h2 class="stage-title">Intervene</h2>
    <span class="stage-time">did not run</span>
    <p class="stage-sub">Change one variable, re-run, read the difference as causal evidence. It does not repair anything.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede lede--skip">Did not run &mdash; no causal experiment was performed.</p>
      <p>Nothing was intervened on to test <em>why</em> the model fails. The repair below was found by
        search and validated statistically, which is a different claim from knowing the cause.</p>
    </div>
    <p>M4 is two stages in the contract, not one. <span class="mono">m4_surgery</span> establishes
      <em>cause</em>; <span class="mono">m4_fix</span> &mdash; the next section &mdash; establishes
      <em>remedy</em>. Only the second ran here.</p>
    <div class="note"><b>Why it is easy to miss.</b> Surgery and M5 share the same machinery, and the
      run log tags which role it played with a <span class="mono">module</span> field. This run&rsquo;s
      single surgery event carries <span class="mono">module&nbsp;=&nbsp;&quot;m5&quot;</span> &mdash; it
      ran as the adjudicator, not as an M4 intervention. So no causal experiment was performed on the
      refuted hypothesis.</div>
  </div>
</section>

<!-- ============ M4-fix ============ -->
<section class="stage" id="m4">
  <div class="stage-head"><span class="stage-code">M4-FIX</span><h2 class="stage-title">Repair</h2>
    <span class="stage-time">Tier ceiling L3a</span>
    <p class="stage-sub">Propose candidate repairs and validate each one paired against the unmodified baseline.</p></div>
  <div class="stage-body">
    <div class="ledewrap">
      <p class="lede lede--good">One repair validated: <span class="key key--good">+7.62 points</span> on held-out cases.</p>
      <p>Seven candidates screened, <span class="key key--bad">five made the model worse</span>. The
        survivor &mdash; telling the model to check each option in a fixed order &mdash; held up on 538
        unseen cases at <span class="key key--good">e&nbsp;=&nbsp;6,281</span>. It also broke 20 cases,
        which the loop flags as a subset-specific fix rather than a general one.</p>
    </div>
    <p>Repair runs independently of whether the diagnosis survived. Seven candidates were screened on the
      exploration split; the best one was then re-run as a paired trial on <b>538 held-out cases</b>.
      The tier ceiling allowed decoding-level interventions (L3a), but no candidate above L2 was proposed
      &mdash; every one tried was a prompt-level change.</p>

    <div class="block"><h3>Screening &mdash; 7 candidates</h3>
      <p class="muted" style="font-size:13px">Selection only. These numbers chose a candidate; they are
        not evidence that it works.</p>
      <div class="scroll"><table>
        <thead><tr><th>Tier</th><th>Candidate</th><th>Verdict</th><th class="num">Repaired</th>
          <th class="num">Broke</th><th class="num">Effect</th><th>Accuracy change</th></tr></thead>
        <tbody>__SEL__</tbody></table></div>
      <div class="note"><b>Five of seven made the model worse.</b> Two of those regressed by more than
        10 points &mdash; asking this model to describe the clip first, or to verify its own answer, costs
        far more than it gains. Only <span class="mono">eliminate_each_option</span> and two weak
        partials came out positive at all.</div>
    </div>

    <div class="block"><h3>Confirmation &mdash; the selected candidate</h3>
      <div class="kv">
        <div><dt>Candidate</dt><dd>eliminate_each_option</dd></div>
        <div><dt>Tier / kind</dt><dd>L1 &middot; prompt template</dd></div>
        <div><dt>Paired cases</dt><dd>__NPAIRS__</dd></div>
        <div><dt>Accuracy</dt><dd>__ACCSHIFT__</dd></div>
        <div><dt>Effect</dt><dd style="color:var(--good)">__CEFF__ pts</dd></div>
        <div><dt>e-value</dt><dd style="color:var(--good)">__EVAL__</dd></div>
      </div>
      <pre class="tmpl">__SUMMARY__</pre>
      <div class="block"><h3>The intervention itself</h3>
        <pre class="tmpl">__TMPL__</pre></div>
      <div class="note note--warn"><b>The loop flags its own result as heterogeneous.</b>
        The fix repaired __NFIX__ cases but broke __NBRK__, which is the signature of a subset-specific
        failure mode rather than a general one. The recommended next step is to re-diagnose on what
        separates the helped cases from the hurt ones and gate the fix on that predicate &mdash; not to
        ship it as-is.</div>
    </div>

    <div class="block"><h3>Case book</h3>
      <p class="muted" style="font-size:13px">__NJOIN__ of the 538 confirmation cases have their clip and
        question available locally, so you can hear what the model heard. The correct option is marked;
        &ldquo;model answered&rdquo; is the repaired prompt&rsquo;s output.</p>
      <div class="filters">
        <button type="button" data-f="all" aria-pressed="true">All __NJOIN__</button>
        <button type="button" data-f="fixed" aria-pressed="false">Repaired __NFIXJ__</button>
        <button type="button" data-f="broken" aria-pressed="false">Broke __NBRKJ__</button>
        <button type="button" data-f="unchanged" aria-pressed="false">Unchanged __NUNCH__</button>
        <span class="count" id="ccount"></span>
      </div>
      <div class="cases" id="cases"></div>
    </div>
  </div>
</section>

</main></div></div>

<footer><div class="wrap">
  <div>Generated from the run&rsquo;s own log and artifacts &mdash; <code>run_log.jsonl</code>,
    <code>explore/exploratory_report.json</code>, <code>logs/report/</code>,
    <code>logs/fixes/02_L1_eliminate_each_option/</code>. Figures are the run&rsquo;s own output, unmodified.</div>
  <div>Model <code>__MODEL__</code> &middot; judge <code>sonnet</code> &middot; EvalVitals
    <code>__VER__</code> &middot; data fingerprint <code>__FP__</code> &middot; audio downsampled to
    16&#8239;kHz mono for embedding.</div>
</div></footer>

<script>
const CASES = __CASES__;
const AUDIO = __AUDIO__;
const grid = document.getElementById('cases');
const countEl = document.getElementById('ccount');
const STATUS = {
  fixed:     {label:'repaired', cls:'chip--good'},
  broken:    {label:'broke',    cls:'chip--bad'},
  unchanged: {label:'unchanged',cls:'chip--mute'}
};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function optionRows(c){
  return c.choices.map(raw => {
    const m = raw.match(/^\s*\(([A-Z])\)\s*(.*)$/s);
    const letter = m ? m[1] : '';
    const text = m ? m[2] : raw;
    const hit = letter && letter === c.expected;
    return `<li class="${hit ? 'is-expected' : ''}"><span class="lt">${esc(letter||'-')}</span><span>${esc(text)}</span></li>`;
  }).join('');
}

function card(c){
  const st = STATUS[c.status];
  const src = AUDIO[c.id];
  return `<article class="case" data-status="${c.status}">
    <div class="case-top">
      <span class="chip ${st.cls}">${st.label}</span>
      <span class="case-task">${esc(c.task||'')}</span>
      <span class="case-dur">${c.duration.toFixed(1)}s</span>
    </div>
    <p class="case-q">${esc(c.instruction)}</p>
    ${src ? `<audio controls preload="metadata" src="${src}"></audio>` : ''}
    <ul class="opts">${optionRows(c)}</ul>
    <div class="case-foot">
      <span class="lbl">Model answered</span>
      <span class="ans">${esc(c.output)}</span>
      <span class="case-id">${esc(c.id.slice(0,8))}</span>
    </div>
  </article>`;
}

function render(filter){
  const rows = filter === 'all' ? CASES : CASES.filter(c => c.status === filter);
  grid.innerHTML = rows.map(card).join('');
  countEl.textContent = `showing ${rows.length} of ${CASES.length}`;
}

document.querySelectorAll('.filters button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(o =>
      o.setAttribute('aria-pressed', String(o === b)));
    render(b.dataset.f);
  });
});
render('all');

/* rail highlight */
const links = [...document.querySelectorAll('.rail-item')];
const obs = new IntersectionObserver(entries => {
  entries.forEach(en => {
    if (!en.isIntersecting) return;
    links.forEach(l => l.classList.toggle('is-active', l.dataset.stage === en.target.id));
  });
}, {rootMargin: '-15% 0px -70% 0px'});
document.querySelectorAll('.stage').forEach(s => obs.observe(s));
</script>
"""

meta = "".join(f"<span>{m}</span>" for m in [
    "MMAU test-mini", "Qwen2-Audio-7B-Instruct", "896 cases", "1 cycle",
    "8 analyzers", "tier ceiling L3a", "24 min wall clock",
])

repl = {
    "__PROTO__": e(d['run']['protocol']),
    "__META__": meta,
    "__TILES__": tiles,
    "__RAIL__": rail,
    "__COV__": cov_rows,
    "__SILENT__": " and ".join(f"<span class='mono'>{e(s)}</span>" for s in silent),
    "__FIG_BALANCE__": figure_block("class_balance", "FAIL / PASS case balance",
                                    readings.get("class_balance", {}).get("reading")),
    "__QUESTION__": e(d['explore']['plain_question']),
    "__TAKE__": takeaways,
    "__FIG_TOP__": charts_lead,
    "__FIG_REST__": charts_rest,
    "__SIGS__": signals,
    "__NADJ__": str(adj['n_rejected']),
    "__NCAND__": str(adj['n_candidates']),
    "__CAVEATS__": caveats,
    "__CRITIQUE__": critique,
    "__CONCLUSION__": e(d['m2']['conclusion']),
    "__NREJ__": str(n_reject),
    "__NSTAT__": str(len(stats)),
    "__STATS_SIG__": stat_rows_sig,
    "__STATS_NULL__": stat_rows_null,
    "__NNULL__": str(n_null),
    "__FIG_M2__": figure_block("m2_effects", "M2 effect sizes across tested signals"),
    "__FM__": e(hyp['failure_mode']),
    "__HYP__": e(hyp['statement']),
    "__M5EFF__": f"{m5r['effect_size']:+.4f}",
    "__M5CI__": f"{m5r['evidence']['ci'][0]:+.3f} .. {m5r['evidence']['ci'][1]:+.3f}",
    "__M5CONF__": f"{m5r['confidence']:.3f}",
    "__M5GRADE__": e(m5r['evidence']['evidence_grade']),
    "__M5VERDICT__": e(m5r['verdict']),
    "__NTESTED__": str(fdr.get('n_tested', '?')),
    "__SEL__": sel_rows,
    "__NPAIRS__": str(cfm['n_pairs']),
    "__ACCSHIFT__": f"{base_acc:.1%} &rarr; {cand_acc:.1%}",
    "__CEFF__": pct(cfm['effect'], 2),
    "__EVAL__": f"{cfm['e_value']:,.0f}",
    "__SUMMARY__": e(cfm['summary']),
    "__TMPL__": e(d['m4']['prompt_template']),
    "__NFIX__": str(cfm['n_fixed']),
    "__NBRK__": str(cfm['n_broken']),
    "__NJOIN__": str(len(cases)),
    "__NFIXJ__": str(n_fixed),
    "__NBRKJ__": str(n_broken),
    "__NUNCH__": str(len(cases) - n_fixed - n_broken),
    "__MODEL__": "qwen2-audio-7b-instruct",
    "__VER__": e(d['run']['evalvitals_version']),
    "__FP__": e(d['run']['data_fingerprint']),
    "__CASES__": json.dumps(cases, ensure_ascii=False),
    "__AUDIO__": json.dumps(audio),
}

out = PAGE
for k, v in repl.items():
    out = out.replace(k, v)

path = Path(args.out)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(out)
print(f"wrote    {path}  ({path.stat().st_size / 1048576:.1f} MB)")
leftover = [k for k in repl if k in out]
if leftover:
    print("! unreplaced tokens:", leftover)
