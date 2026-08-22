# Five-paper diagnosis benchmark

This example stress-tests EvalVitals with five deliberately different research
papers. It is a **framework diagnosis benchmark**, not a claim that five papers
prove a repair works for a target model. Its purpose is to expose where the
framework overstates evidence, confuses benchmark artefacts with model health,
or proposes an intervention beyond the evidence.

The corpus covers factuality, multi-metric evaluation, black-box hallucination
detection, synthetic-versus-real distribution shift, and agent/tool-use
failures. Each checked-in entry in [`papers.json`](papers.json) includes a
canonical source URL and a stress test the final diagnosis must address.

## Data policy

Only scripts and paper metadata are versioned. PDFs, extracted page text, and
all results are local files ignored by [`.gitignore`](.gitignore). Download
from the canonical source and comply with each source's terms before use.

## Run

From this directory:

```bash
# PDF extraction is the only extra requirement for this example.
pip install evalvitals pypdf

# 1. Fetch five PDFs into ignored data/papers/.
python download_papers.py

# 2. Produce ignored, page-level evidence records.
python build_records.py

# 3. Use the public EvalVitals interface to analyze the corpus.
#    The selected coding-agent CLI must already be installed and authenticated.
python run_benchmark.py --backend codex
```

The final command is intentionally just an adapter over the public interface:

```bash
python -m evalvitals.cli explore data/paper_records.jsonl --backend codex ...
```

Its output goes to `outputs/five_paper_diagnosis/`, including the exploratory
report, generated analysis code, evidence tables, and candidate hypotheses.
No source PDF or result artifact is committed.

## What to assess

For each paper, the analysis must return an evidence-grounded diagnosis that
identifies:

1. The reported failure and its measured outcome.
2. The evidence and its scope (paper id and page references).
3. Confounders or evaluation threats that prevent a causal claim.
4. The least-invasive repair worth testing, plus its expected side effects.
5. A validation design and the conditions under which the right outcome is
   **inconclusive** rather than “fixed.”

Then review the cross-paper output for framework gaps. A good benchmark result
is allowed to reject a paper-level repair recommendation; it should never turn
a literature summary into proof that a deployed model improved.

## Extending the corpus

Add a manifest row with `id`, `title`, `year`, `source_url`, `pdf_url`,
`diagnosis_axis`, and `expected_stress_test`. Keep paper files in `data/`; do
not add PDFs, extracted JSONL, or generated outputs to git.

## Intervention-paper pilot

[`intervention_papers.json`](intervention_papers.json) is a second, executable
five-paper protocol. It replaces a literature-only comparison with five papers
that name an intervention family: zero-shot chain-of-thought,
self-consistency, least-to-most prompting, self-refine, and
chain-of-verification. The first four use a frozen GSM8K slice; the final one
uses a frozen, shuffled TruthfulQA MC1 slice. This is explicitly a small
transfer evaluation, not an exact reproduction of each paper's original model,
prompt, or sample size.

```bash
# Downloads Hugging Face data under ignored data/intervention_pilot/.
python download_benchmark_data.py

# Diagnosis → candidate selection → untouched confirmation.
# The default targets the local OpenAI-compatible Qwen endpoint on port 8010.
# Use all 120 frozen examples: 24 diagnosis, 36 selection, 60 confirmation.
python run_intervention_pilot.py zero_shot_cot --limit 120
python run_intervention_pilot.py self_consistency --limit 120
python run_intervention_pilot.py least_to_most --limit 120
python run_intervention_pilot.py self_refine --limit 120
python run_intervention_pilot.py cove --limit 120
```

The candidate is selected only on the selection split and is then re-run once
on the untouched confirmation split. A confirmation report calls a repair
validated only when that pre-selected candidate clears the paired gate there.
The runner records OpenAI-compatible `finish_reason`. When failed responses
explicitly end with `length`, EvalVitals may test the least invasive L0 runtime
repair: a bounded increase of `max_tokens`. It never infers truncation merely
because an answer looks short. Use `--baseline-max-tokens` to reproduce the
deployment configuration being audited.

The default uses one generated L1 and one generated declarative L2 candidate
per non-L0 run, so it is a bounded screen. Use `--auto-candidates 3` only for
the full candidate family; e-BH is applied over every candidate tested in the
selection phase.

The runner reserves neither paper methods nor auto-fix candidates as proof of
success: it records the paired validation outcome and reports an underpowered
result rather than calling a small-sample gain a validated repair.
