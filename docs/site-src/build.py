#!/usr/bin/env python3
"""Build the EvalRx landing page.

Produces docs/index.html: a complete, standalone HTML5 document for GitHub
Pages, with fonts and OFL license texts referenced as external files under
docs/assets/fonts/ (cacheable, no inline base64 payload).

The terminal walkthrough in the "What a full pass looks like" section is
server-rendered here (TERM_LINES below), not built by JS innerHTML — the
in-page script only staggers each line's reveal. With JS disabled, every
line is already present in the raw HTML.

Usage:
    python3 docs/site-src/build.py
"""
from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

SRC = Path(__file__).parent
DOCS = SRC.parent

SITE_URL = "https://evalvitals.github.io/evalrx/"
TITLE = "EvalRx — diagnose why a model fails, then verify the fix"
DESCRIPTION = (
    "EvalRx is a self-improving loop that probes an open-weight model, "
    "diagnoses the mechanism behind its failures, verifies the diagnosis on "
    "held-out cases, and builds a repair validated against the unmodified "
    "baseline."
)
# Stethoscope: literal to the "model health" framing this page opens with.
FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
    "%3Ctext y='.9em' font-size='90'%3E%F0%9F%A9%BA%3C/text%3E%3C/svg%3E"
)

FONT_FILES = {
    "FRAUNCES_HERO": "fraunces_hero_900.woff2",
    "FRAUNCES_H2": "fraunces_h2_600.woff2",
    "FRAUNCES_ITALIC": "fraunces_italic_500.woff2",
    "PLEXSANS": "plexsans_var.woff2",
    "PLEXMONO_400": "plexmono_400.woff2",
    "PLEXMONO_500": "plexmono_500.woff2",
    "PLEXMONO_600": "plexmono_600.woff2",
}
OFL_FILES = ["OFL-fraunces.txt", "OFL-ibmplexsans.txt", "OFL-ibmplexmono.txt"]

# (html, delay_ms_before_reveal). Above the divider: real control flow on the
# committed deco_hallu_explore / repair-ladder walkthrough. Below it, in
# violet (.c-proj): a projection of one escalated fine-tune — not a measured
# run — clearly labeled as such both here and in the section copy.
TERM_LINES = [
    ('<span class="c-dim">$</span> <span class="c-cmd">evalrx investigate</span> qwen3-vl-8b-instruct <span class="c-dim">\\</span>', 90),
    ('      <span class="c-dim">--probe</span> pope-adversarial <span class="c-dim">--holdout</span> 0.4 <span class="c-dim">--max-tier</span> L4', 260),
    ("", 120),
    ('<span class="c-key">M1</span>  probe          <span class="c-dim">············</span> 606 cases · <span class="c-no">126 fail</span> (20.8%)', 340),
    ('<span class="c-key">M2</span>  explore        <span class="c-dim">············</span> 23 figures · 4 candidate signals', 340),
    ('<span class="c-key">M3</span>  diagnose       <span class="c-dim">············</span> 2 falsifiable hypotheses', 340),
    ('<span class="c-key">M4</span>  verify         <span class="c-dim">············</span> held-out n=242 · e-BH α=0.05', 380),
    ('    <span class="c-ok">✓</span> peaked_attention  <span class="c-dim">stat + protocol: SUPPORTED</span>  CI +0.360..+0.589', 200),
    ('    <span class="c-no">✗</span> top1_share_high   <span class="c-dim">refuted</span>', 120),
    ('    <span class="c-no">✗</span> peripheral_attn   <span class="c-dim">refuted</span>', 120),
    ("", 200),
    ('<span class="c-key">M5</span>  intervene      <span class="c-dim">············</span> peaked_attention Δ +14.2pp <span class="c-dim">(causal, not yet a fix)</span>', 300),
    ('    <span class="c-dim">└</span> repair <span class="c-dim">— climbing the ladder, ceiling L4</span>', 320),
    ('    <span class="c-dim">L1 </span> prompt rewrite        <span class="c-no">✗</span> +0.4pp  <span class="c-dim">p=0.41 vs baseline</span>', 420),
    ('    <span class="c-dim">L2 </span> attention-guided crop <span class="c-no">✗</span> +1.2pp  <span class="c-dim">p=0.19 vs baseline</span>', 420),
    ('    <span class="c-dim">L3b</span> attention reweight    <span class="c-no">✗</span> +1.1pp  <span class="c-dim">p=0.22 vs baseline</span>', 420),
    ('        <span class="c-warn">all candidates exhausted below L4</span>', 300),
    ("", 180),
    ('<span class="c-warn">↑</span>  escalate <span class="c-dim">→</span> <span class="c-key">L4 parameter space</span>', 380),
    ('    fine-tune recipe written <span class="c-dim">→</span> finetune_spec.json', 300),
    ('      base     Qwen3-VL-8B-Instruct <span class="c-dim">(open weights)</span>', 160),
    ('      data     1,204 pairs drawn from the confirmed mode', 160),
    ('      method   LoRA r=16 · 2 epochs · bf16', 160),
    ('      re-test  242 held-out cases vs unmodified baseline', 160),
    ("", 240),
    ('<span class="c-key">↺</span>  L4 executor available <span class="c-dim">—</span> plain LoRA on the LLM ships today; this walkthrough continues below as a projection.', 220),
    ("", 320),
    ('<span class="c-proj">┄┄┄ below this line: projected on target hardware, not measured ┄┄┄</span>', 480),
    ("", 200),
    ('<span class="c-key">P?</span>  strategy selection <span class="c-dim">— confirmed mode: over-concentrated attention</span>', 300),
    ('    <span class="c-ok">▸</span> P1 attention-grounding LoRA  <span class="c-dim">targets the measured mechanism</span>', 200),
    ('    <span class="c-ok">▸</span> P2 contrastive DPO           <span class="c-dim">hallucinated yes / correct no</span>', 200),
    ('    <span class="c-dim">· P3 counterfactual SFT        held in reserve</span>', 160),
    ('    <span class="c-dim">· P4 full fine-tune            held in reserve</span>', 260),
    ("", 200),
    ('<span class="c-key">$$</span>  cost estimate <span class="c-dim">— 8 × H200 SXM @ $2.30/GPU-h</span>', 320),
    ('    P1  LoRA        0.8 h    <span class="c-proj">$14.72</span>', 180),
    ('    P2  DPO         2.4 h    <span class="c-proj">$44.16</span>', 180),
    ('    re-test         0.3 h    <span class="c-proj">$5.52</span>', 180),
    ('    <span class="c-dim">──────────────────────────────</span>', 120),
    ('    total           3.5 h    <span class="c-proj">$64.40</span>', 380),
    ("", 220),
    ('<span class="c-key">▶</span>  projected run <span class="c-dim">— 8 × H200, bf16, grad-ckpt</span>', 320),
    ('    <span class="c-dim">P1</span>  step  200/1200  loss 0.842  <span class="c-dim">▓▓▒░░░░░░░</span>  gpu 91%', 260),
    ('    <span class="c-dim">P1</span>  step  700/1200  loss 0.517  <span class="c-dim">▓▓▓▓▓▓░░░░</span>  gpu 93%', 260),
    ('    <span class="c-dim">P1</span>  step 1200/1200  loss 0.388  <span class="c-dim">▓▓▓▓▓▓▓▓▓▓</span>  <span class="c-ok">converged</span>', 300),
    ('    <span class="c-dim">P2</span>  step  900/900   loss 0.204  <span class="c-dim">▓▓▓▓▓▓▓▓▓▓</span>  <span class="c-ok">converged</span>', 340),
    ("", 220),
    ('<span class="c-key">↺</span>  re-test <span class="c-dim">— 242 held-out cases vs unmodified baseline</span>', 340),
    ('    hallucination rate  34.4% <span class="c-dim">→</span> <span class="c-ok">21.9%</span>   <span class="c-ok">Δ −12.5pp</span>', 260),
    ('    McNemar paired      <span class="c-ok">p = 0.0007</span>  <span class="c-dim">repair holds</span>', 240),
    ('    no-free-lunch check <span class="c-dim">present-object detection 100% → 99.6%</span>', 240),
    ("", 240),
    ('<span class="c-proj">   Figures above are a projection of what one escalated</span>', 140),
    ('<span class="c-proj">   investigation would cost and produce on this ladder path. Running</span>', 140),
    ('<span class="c-proj">   it end-to-end through today\'s generic LoRA executor is the gap.</span>', 0),
]


def render_term_lines() -> str:
    parts = []
    for content, delay in TERM_LINES:
        parts.append(f'<span class="l" data-t="{delay}">{content or " "}</span>')
    return "".join(parts)


def build_head() -> str:
    og_title = html.escape(TITLE)
    og_desc = html.escape(DESCRIPTION)
    return f"""<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{og_title}</title>
<meta name="description" content="{og_desc}">
<meta name="robots" content="index, follow">
<meta name="keywords" content="LLM evaluation, failure analysis, model diagnosis, interpretability, VLM, hallucination detection, model repair, open-weight models, evalrx">
<link rel="canonical" href="{SITE_URL}">
<link rel="icon" href="{FAVICON}">

<meta property="og:type" content="website">
<meta property="og:site_name" content="EvalRx">
<meta property="og:title" content="{og_title}">
<meta property="og:description" content="{og_desc}">
<meta property="og:url" content="{SITE_URL}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{og_title}">
<meta name="twitter:description" content="{og_desc}">
"""


def main() -> int:
    template = (SRC / "template.html").read_text(encoding="utf-8")

    body = template
    for token, fname in FONT_FILES.items():
        body = body.replace(f"__FONT_{token}_SRC__", f"url(assets/fonts/{fname}) format('woff2')")
    body = body.replace("__TERM_LINES__", render_term_lines())

    head = build_head()
    doc = "<!doctype html>\n" f'<html lang="en">\n<head>\n{head}</head>\n<body>\n{body}\n</body>\n</html>\n'

    out = DOCS / "index.html"
    out.write_text(doc, encoding="utf-8")

    assets_out = DOCS / "assets" / "fonts"
    assets_out.mkdir(parents=True, exist_ok=True)
    for fname in FONT_FILES.values():
        shutil.copyfile(SRC / "assets/fonts" / fname, assets_out / fname)
    for fname in OFL_FILES:
        shutil.copyfile(SRC / "assets/fonts" / fname, assets_out / fname)

    # .nojekyll, robots.txt and sitemap.xml are NOT written here in this repo:
    # this page rides evalrx's mkdocs gh-deploy (see docs/../mkdocs.yml), which
    # writes its own .nojekyll and generates a real sitemap.xml covering every
    # doc page, not just this one URL. Writing them here would only race or
    # get silently overwritten by that build.

    kb = out.stat().st_size / 1024
    print(f"built {out}  ({kb:.1f} KB)")
    print(f"title: {TITLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
