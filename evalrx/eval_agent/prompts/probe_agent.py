"""Prompt templates for M1 probe-agent analyzer selection."""

_SELECTION_PROMPT_TMPL = """\
You are selecting diagnostic analyzers for a model evaluation experiment.
Choose the analyzers that would surface the most useful evidence given the \
researcher's description.

EXPERIMENT DESCRIPTION:
{description}
{task_domain_section}
{success_criteria_section}
{failure_patterns_section}{cases_section}
MODEL TYPE: {model_kind}

AVAILABLE ANALYZERS ({n_available} compatible with this model):
{analyzer_list}
{failed_section}{prior_hypotheses_section}
Select up to {max_n} analyzers. Return ONLY a JSON object, no other text:
{{"analyzers": ["name1", "name2", ...], "rationale": "one sentence explaining the selection"}}

If the analyzers you can actually rely on (excluding any listed as failed)
cannot adequately surface the suspected mechanism, additionally include
"need_custom": "<one line describing the probe to generate>" in the JSON —
a bespoke probe will be code-generated and run in a sandbox.
"""
