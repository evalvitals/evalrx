"""Prompt templates and prompt policies for exploratory analysis."""

from __future__ import annotations

from evalvitals.agent_runtime.skills.prompt_policy import fences_hint, skills_hint

RECORDS_FILENAME = "records.json"  # also read by explore_run.py / dashboard_app.py
_RESULT_MARKER = "EXPLORATORY_RESULT_JSON="

_INTRO_AND_QUESTION = """\
You are an exploratory data-analysis agent (Lambda-style): given ANY tabular
dataset — with a binary outcome, a multi-class outcome, a continuous outcome,
or no outcome at all — you write Python that discovers and charts the
structure that actually matters, adapting the story to what the data is.

Question:
{question}

"""

_DATA_ACCESS_RECORDS = """\
A JSON file named "{input_filename}" is in the current working directory.
It contains a list of row dictionaries. Data profile (the host already
inferred column roles/dtypes and classified the outcome below — trust this
over re-guessing from raw values):
{data_profile}

Write a self-contained Python script that performs a THOROUGH, Lambda-style
exploratory analysis and PRODUCES A RICH SET OF CHARTS BY DEFAULT.

Setup:
- reads "{input_filename}" from the current working directory
- may use only local Python packages (pandas / numpy / matplotlib are fine); no
  network and no repo mutation
- the profile's "columns" block gives each column's role (id / outcome / group
  / time / predictor) and dtype; its "outcome" block ({{"present","column","kind"}})
  tells you whether there is one and what kind it is. Do not invent an outcome
  or a FAIL/PASS split when "outcome.present" is false.

{framing}

"""

_DATA_ACCESS_RAW_FOLDER = """\
The RAW M1-style output is at "{raw_input_dir}/" in the current working
directory — nothing has been parsed, reshaped, or profiled for you. A
filesystem-level scan of what's on disk (file/dir counts, extensions, a
sample of entries — NOT parsed rows):
{folder_scan}

Write a self-contained Python script that performs a THOROUGH, Lambda-style
exploratory analysis and PRODUCES A RICH SET OF CHARTS BY DEFAULT.

Setup:
- FIRST, load and organize the raw data into ONE tidy table YOURSELF — this is
  part of your job, not a solved input. The files under "{raw_input_dir}/" may
  be a single file or many, and each file's JSON shape is not guaranteed: it
  could already be a flat list of row dicts, JSONL (one JSON object per line),
  or a dict holding scalar run/file metadata (e.g. "model", "seed") alongside
  a list of per-case records under a conventional key such as "cases" /
  "results" / "rows" / "items" / "data" / "examples" / "samples" — or
  something else entirely. Inspect what is actually there before assuming a
  shape. If a file wraps a list of per-case dicts alongside scalar metadata,
  merge that metadata into every row it produces (e.g. a per-model file's
  "model" field becomes a "model" column on each of that file's rows).
- a browser workbench bundle may also contain ``records.json`` (normalized
  tabular rows), ``media_units.jsonl`` (file/page/time-segment metadata), and
  a ``media/`` tree holding original images, PDFs, audio, or video. Start from
  the normalized rows when present, join media units by source path when useful,
  and inspect representative media only when your CLI can actually read it.
  Do not invent semantic content for audio/video/PDF pages you could not open;
  state the capability limitation in a caveat instead.
{outcome_hint}- after building the tidy table, determine whether it has a recognizable
  outcome column (binary / categorical / continuous) or none at all — do NOT
  invent a FAIL/PASS split when there is no such column.
- also write the tidy table you built to "{input_filename}" (a JSON list of
  row dicts) in the current working directory, so it is available to whoever
  reviews this analysis afterward.
- may use only local Python packages (pandas / numpy / matplotlib are fine); no
  network and no repo mutation

{framing}

"""

_GENERIC_FRAMING = """\
OUTCOME FRAMING — apply whichever case matches what you find after loading:
  - BINARY outcome (2 distinct values in some column): call the two groups
    FAIL and PASS and tell the FAIL-vs-PASS story — class balance; per numeric
    signal vs FAIL/PASS (distribution view + binned fail-rate curve); a ranked
    bar of each signal's FAIL-vs-PASS separation; fail rate by categorical
    group columns; signal correlations; 1-2 scatter plots of the most
    discriminative pairs coloured by outcome.
  - CATEGORICAL outcome (3+ classes): tell the per-class story — do NOT
    collapse it into a binary split. Class balance per class; per numeric
    signal's distribution across classes (box/violin) plus a ranked
    cross-class separation bar; class composition by categorical group
    columns; signal correlations; 1-2 scatter plots coloured by class.
  - CONTINUOUS outcome: a correlation/regression-style story — do not
    binarize it unless the question explicitly asks for that. Outcome
    distribution; per numeric signal vs outcome (scatter + trend line, plus a
    binned mean-outcome curve); a ranked bar of correlation magnitude with the
    outcome; outcome distribution by categorical group columns; predictor
    correlation heatmap; 1-2 scatter plots of the most associated pairs.
  - NO recognizable outcome/target column: unsupervised exploration — describe
    structure, do NOT invent a label that isn't in the data. Missingness
    overview; per-numeric-column distributions; per-categorical-column value
    counts (skip columns with very high cardinality as a bar chart, note it in
    caveats); correlation table/heatmap; 1-2 scatter plots of the most
    correlated numeric pairs; if a group/time column exists, contrast numeric
    distributions across it."""

_ANALYSIS_CONTRACT = """\
VISUAL ANALYSIS — before writing plotting code, make an explicit intermediate
visualization plan. The plan is part of the machine-readable output and should
show that YOU selected plot types from the data semantics, not from a fixed
template. Aim for 6-12 charts/plots that together tell the dataset's story.

First build a "visual_plan" list. Each item should be a dict:
  {{
    "name": "<stable artifact/chart name>",
    "display_name": "<short human title, no raw generated/probe id>",
    "question": "<what this visual answers>",
    "data_shape": "<numeric-vs-binary | numeric-vs-categorical | numeric-vs-numeric | many-numeric | paired | unsupervised | ...>",
    "plot_kind": "<chosen plot type, e.g. bar, line, scatter, box, violin, heatmap, paired_slope>",
    "fallback_kind": "<bar|line|scatter when a deterministic host chart is useful>",
    "required_columns": ["..."],
    "rationale": "<why this plot type fits the data and avoids misleading summaries>",
    "disposition": "<primary|supporting|skipped>",
    "not_promoted_reason": "<required when supporting; why this is context/diagnostic material rather than a ranked takeaway>"
  }}

Use these decision principles:
  - categorical/binary outcome: rate/count bar with n annotated in the table.
  - numeric predictor vs categorical/binary outcome: prefer distribution views
    (box/violin/strip) when writing rich PNG plots; include a deterministic
    summary chart only when useful.
  - binned numeric trend (event rate, or mean of a continuous outcome): line
    over ordered bins/percentiles.
  - numeric vs numeric: scatter, optionally colored/stratified by outcome or group.
  - many numeric signals: ranked effect/association bar plus correlation heatmap.
  - paired/intervention data: paired slope or discordant-count visual.
  - no outcome column: prioritize distributions, missingness, correlation
    structure, and group contrasts over any label-vs-label story.
  - skip a planned visual when required columns are absent or sample size makes it
    misleading; say so in caveats.
  - A ``primary`` visual MUST be cited by at least one takeaway's ``chart_names``.
    Use ``supporting`` only for context/diagnostic material and give a concrete
    ``not_promoted_reason``. Do not emit an artifact for a ``skipped`` visual.

For EVERY chart you report in "charts":
- write its plotted data as a CSV under "tables/<name>.csv"
- add a spec {{"name","display_name","kind","data","x","y","title"}} with data="tables/<name>.csv"
  and kind in {{"bar","line","scatter"}}. The HOST renders these deterministically,
  so PRE-AGGREGATE distributions into the CSV (histogram = bin->count; outcome
  rate or mean-outcome curve = bin->value; group comparison = group->value) —
  never rely on a raw dump.
ADDITIONALLY you MAY draw richer figures (box / violin / heatmap / scatter-matrix)
directly as PNG under "figures/" and list them in "plots"; a figure-styling skill
(when available) will make these publication-quality.

This is PURE EXPLORATORY DATA ANALYSIS. Describe what the data shows. Do NOT
propose causal explanations, do NOT claim anything is "confirmed" or
"validated", and do NOT frame findings as hypotheses to be tested — hypothesis
generation and validation are a different, separate step that this tool does
not perform. Stick to descriptive, evidence-grounded statements.

Plain-language framing of the question (the dashboard's page headline, so it
renders BEFORE any takeaway):
- "plain_question": restate the Question above in ONE everyday sentence — what
  is actually being investigated, in plain words a non-technical reader would
  understand, not a compressed research-question shorthand. The caller's own
  question may itself be dense/technical (column names, stats jargon, a
  numbered list of sub-asks); your job is to say in plain terms what it's
  really asking, not to copy it. Same jargon/acronym/symbol ban as
  "plain_title" below, and it is checked the same way.

Reader-friendly writing contract:
- Write for a user who may understand the evaluation problem but may NOT know
  machine-learning or statistics terms. Prefer concrete, plain language over
  terms like "rank-biserial", "AUC", "VIF", "collinearity", "effect size",
  "bootstrap", or "regression" in reader-facing titles and summaries. If one of
  those terms is unavoidable, define it immediately in the same sentence.
- Separate three ideas in every reader-facing explanation: what was observed,
  why it matters for the user's question, and what remains uncertain.
- Use raw counts alongside percentages whenever you can, e.g. "24 of 30 cases
  (80%)" rather than only "80%". For group comparisons, name both groups and
  their sample sizes.
- Never imply that an observed relationship proves a cause. Use wording like
  "is linked with", "appears more often with", or "is a candidate signal", not
  "drives", "causes", or "explains", unless an intervention actually tested it.
- Clearly flag weak evidence in ordinary language: small samples, missing data,
  uneven group sizes, repeated cases, class imbalance, wide uncertainty, or
  thresholds chosen on the same exploration rows.


Takeaways (THE PRIMARY OUTPUT — this is what a reader sees first):
- "takeaways": a ranked list of 4-8 dicts, most important/surprising finding
  first, each shaped exactly like:
    {{"plain_title": "<the SAME finding as 'title' below, restated in ONE
                       everyday sentence a non-technical reader (a PM, not a
                       statistician) can understand at a glance. NO acronyms
                       (AUC, ROC, CI, ECDF...), NO statistics jargon
                       (collinear, logistic, coefficient, p-value, quantile,
                       monotonic, confound, variance, latent, mediator...),
                       NO symbols (→, ρ, σ). Numbers/percentages ARE
                       encouraged — plain language does not mean vague; keep
                       the evidence, drop the notation. This is the headline
                       a reader sees first; it is checked automatically and a
                       jargon-y or copy-pasted one is sent back for a
                       rewrite.>",
      "title": "<one concise companion sentence with the precise technical
                 detail and real counts/percentages; statistics terms are fine
                 here when they add useful precision>",

      "chart_names": ["<name(s) from 'charts' or 'plots' that support it>"],
      "table_names": ["<key(s) from 'tables' that support it, if any>"],
      "analysis": "<2-4 plain-language sentences: (1) what was observed, (2)
                    why it matters for the user's question, and (3) what
                    remains uncertain. Cite actual counts, percentages,
                    columns, and groups behind the chart(s).>",
      "caveat": "<short plain-language note on what this does NOT show, or '' if nothing notable>"}}
  EVERY important chart/plot you produce should be referenced by at least one
  takeaway's "chart_names" — never leave a chart orphaned with no explanation,
  and never write a takeaway with no supporting chart/table unless the data
  genuinely has none to offer (rare).

Report/dashboard contract:
- Emit ONE "dashboard_storyboard" panel dict (a list with one entry) orienting
  the reader on the data/question before the takeaways:
    {{"id": "problem_setting", "title": "Problem Setting", "summary": "...",
      "items": ["..."], "artifact_refs": ["data_profile"]}}
  The problem-setting summary/items must state, in plain language:
    * what is being studied;
    * the user's question;
    * the unit of analysis (one row = one case/batch/request/etc.);
    * how many records were loaded and which groups/splits are relevant;
    * what the outcome column means in practical terms (e.g. what FAIL/PASS
      mean, or what a continuous outcome measures).
  Do not add "analysis" or "hypotheses" panels — "takeaways" already covers
  that ground, and this tool does not generate hypotheses.

Secondary fields (for programmatic consumers such as a downstream confirmatory
pipeline — NOT the primary reader-facing narrative; keep these terse):
- add "chart_readings": one short dict for EVERY emitted chart or plot, keyed
  by its exact chart name or PNG stem, with
  {{"chart": "<name/title>", "reading": "<what a human should see>",
  "do_not_infer": "<what this chart cannot prove>"}}. This is the explanation
  displayed immediately below that visual; never leave it generic or omit it.
  The reading should explain the visible pattern in one or two simple
  sentences, name the comparison groups and sample sizes when available, and
  avoid merely restating the axes. The do_not_infer field should name the main
  boundary, such as "does not prove causation", "small group", "chosen on
  exploration rows", or "missing data may affect this comparison".
- add "claims" only for carefully worded descriptive/confirmable statements. Each
  claim must cite chart/signal identifiers in "evidence_ids"; set status to
  "descriptive" (never "supported" — this tool does not confirm anything).
- add "critique": agent self-audit notes about leakage, small n, double-dipping,
  missingness, misleading plot choices, or alternative explanations.
- Never use raw internal IDs like "generated_probe1_false_detection" as user-facing
  chart titles or claim text. Use display names such as "Sanity check: probe
  false-detection flag", and demote probe-derived fields to sanity-check
  evidence rather than explanatory findings.
- PREFERRED: for any composite / threshold / interaction signal that is a
  DETERMINISTIC FUNCTION of the numeric columns, attach a "recipe" so a
  downstream confirmatory pipeline can compute it on a HELD-OUT split later
  (this tool itself does not run that confirmation):
    "recipe": {{"name": "<new signal key>", "kind": "expr",
                "expr": "<boolean/numeric expression over the numeric columns above>"}}
  The expr may use the columns BY NAME, comparisons (< <= > >= == !=), and/or/not,
  arithmetic (+ - * / %), and abs/min/max/float/int/len. It must NOT reference the
  label/outcome column (a recipe is a PREDICTOR, never the answer). Example:
    "recipe": {{"name": "small_and_peripheral", "kind": "expr", "expr": "(obj_size < 40) and (focus_share < 0.3)"}}
  Emit a recipe rather than prose whenever the candidate is computable from the columns.
- ALTERNATIVELY, you MAY attach host-adjudicable "sufficient" statistics computed
  from the rows, as ONE of these shapes:
    {{"kind": "two_group", "a": [0/1, ...], "b": [0/1, ...]}}   # is_fail indicators among signal-ABSENT (a) vs signal-PRESENT (b) cases
    {{"kind": "paired_binary", "b": <int>, "c": <int>}}          # discordant counts of a paired intervention (b = flips the good way, c = the bad way)
  Do NOT emit "reject"/"e_value"/"p_value" anywhere — this tool never adjudicates
  a verdict itself; a self-declared verdict is ignored. Omit both for
  descriptive-only signals.
- prints the final result as the LAST stdout line exactly like (note "charts" is a
  RICH list here, one entry per CSV you wrote). The example below illustrates the
  JSON SHAPE using a binary-outcome dataset; KEEP THE SAME KEYS but replace the
  FAIL/PASS/fail_rate wording with language that matches the ACTUAL outcome kind
  from the profile above (categorical classes, a continuous outcome's mean/curve,
  or plain unsupervised structure when there is no outcome):
  {marker}{{
    "plain_question": "Whether small objects are harder for the model to get right.",
    "observations": ["..."],
    "visual_plan": [
      {{"name": "failrate_by_objsize",
        "display_name": "Failure rate by object size",
        "question": "Does object size change failure risk?",
        "data_shape": "numeric-vs-binary",
        "plot_kind": "line",
        "fallback_kind": "line",
        "required_columns": ["obj_size", "label"],
        "rationale": "Ordered bins show risk trend without assuming linearity.",
        "disposition": "primary",
        "not_promoted_reason": ""}}
    ],
    "takeaways": [
      {{"plain_title": "Small objects trip up the model much more often than big ones.",
        "title": "Small-object cases failed in 18 of 100 rows (18%), compared with 4 of 100 larger-object rows (4%).",

        "chart_names": ["failrate_by_objsize"],
        "table_names": [],
        "analysis": "The chart compares small-object cases with larger-object cases. Small-object cases failed in 18 of 100 rows (18%), compared with 4 of 100 larger-object rows (4%). This matters because it points to a concrete group of cases to inspect next, but the chart does not show whether object size itself is the reason for the failures.",
        "caveat": "Descriptive only — other differences between the two groups may be responsible, and this is not a causal finding."}}
    ],
    "chart_readings": [
      {{"chart": "failrate_by_objsize",
        "reading": "The smallest object-size bins have visibly higher failure rates than the larger-object bins. Include the number of cases in each bin so the reader can see whether the comparison is based on enough data.",
        "do_not_infer": "This chart does not prove object size causes the error; it only shows where failures are more common in these rows."}}
    ],
    "claims": [
      {{"id": "C1",
        "text": "Small object size is a descriptive failure correlate.",
        "status": "descriptive",
        "evidence_ids": ["chart:failrate_by_objsize"],
        "interpretation": "A candidate signal for downstream confirmatory analysis.",
        "do_not_infer": "Not causal; not yet confirmed by any statistical test."}}
    ],
    "dashboard_storyboard": [
      {{"id": "problem_setting", "title": "Problem Setting",
        "summary": "This run studies labeled model-evaluation cases to find which recorded signals are linked with failures.",
        "items": ["User question: which case features are linked with FAIL outcomes?", "Unit of analysis: one row is one evaluated case.", "Outcome: FAIL means the model gave the wrong answer; PASS means it answered correctly.", "Report the number of loaded rows and the FAIL/PASS counts here."],
        "artifact_refs": ["data_profile"]}}
    ],
    "candidate_signals": [
      {{"name": "...", "display_name": "<human-readable signal label>",
        "rationale": "...", "suggested_test": "...",
        "recipe": {{"name": "...", "kind": "expr",
                   "expr": "(col_a < 40) and (col_b < 0.3)"}}}}
    ],
    "plots": ["figures/corr_heatmap.png"],
    "tables": {{}},
    "charts": [
      {{"name": "class_balance", "kind": "bar",
        "display_name": "FAIL/PASS case balance",
        "data": "tables/class_balance.csv", "x": "outcome", "y": "count",
        "title": "FAIL vs PASS"}},
      {{"name": "failrate_by_objsize", "kind": "line",
        "display_name": "Failure rate by object size",
        "data": "tables/failrate_by_objsize.csv", "x": "obj_size_bin",
        "y": "fail_rate", "title": "Fail rate by object size"}},
      {{"name": "top_discriminators", "kind": "bar",
        "data": "tables/top_discriminators.csv", "x": "signal",
        "y": "separation", "title": "Top FAIL/PASS discriminators"}}
    ],
    "caveats": ["..."],
    "critique": ["..."],
    "recommended_confirmatory_tests": ["..."]
  }}
{skills_hint}
Return ONLY the Python code{fences_hint}."""

_GENERATE_PROMPT_RECORDS = _INTRO_AND_QUESTION + _DATA_ACCESS_RECORDS + _ANALYSIS_CONTRACT
_GENERATE_PROMPT_RAW_FOLDER = _INTRO_AND_QUESTION + _DATA_ACCESS_RAW_FOLDER + _ANALYSIS_CONTRACT

_REPAIR_PROMPT = """\
The exploratory analysis script failed or produced an invalid result.

Question:
{question}

Data profile:
{data_profile}

Previous code:
```python
{code}
```

Sandbox stdout:
{stdout}

Sandbox stderr:
{stderr}

Parser/execution error:
{error}

Rewrite the script so it succeeds end-to-end and still prints a final
{marker} JSON line with the required keys, in the same format as before.
Return ONLY Python code{fences_hint}."""

RESULT_MARKER = _RESULT_MARKER
GENERIC_FRAMING = _GENERIC_FRAMING
GENERATE_PROMPT_RECORDS = _GENERATE_PROMPT_RECORDS
GENERATE_PROMPT_RAW_FOLDER = _GENERATE_PROMPT_RAW_FOLDER
REPAIR_PROMPT = _REPAIR_PROMPT

_fences_hint = fences_hint
_skills_hint = skills_hint
