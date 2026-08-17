"""Prompt templates for fix candidate generation and repair."""

_L1_PROMPT = """\
You are designing PROMPT-LEVEL fixes (tier L1: the input space only) for a \
model failure. The model may be text-only or vision-language.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

Propose up to {k} prompt rewrite strategies that could repair these failures
WITHOUT changing the model or adding pipeline steps.  Each strategy is a
template applied to every case prompt; it MUST contain the literal placeholder
{{prompt}}.

Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>", "prompt_template": "<template with {{prompt}}>"}}]"""

_L2_PROMPT = """\
You are designing SCAFFOLD-LEVEL fixes (tier L2: a pipeline around the \
unchanged model) for a model failure. The model may be text-only or
vision-language.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

OPTIONAL IMAGE TOOLS (use only when the case has an image; text-only cases
should use prompt rewrites and/or repeated sampling instead):
{catalog}

Propose up to {k} pipelines. Each may chain image tools, rewrite the prompt
(template MUST contain {{prompt}}), use bounded decoding controls in
generation_kwargs (max_tokens, temperature, top_p, stop), sample the model
n_samples times, or select one reviewed multi-call strategy: direct,
least_to_most, self_refine, chain_of_verification. For structured answer tasks,
output_key_pattern may be a regex with one capture group used for answer-only
voting; it must not contain gold answers. Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>",
   "image_ops": [{{"tool": "<catalog name>", "params": {{}}}}],
   "prompt_template": "{{prompt}}", "n_samples": 1,
   "generation_kwargs": {{}}, "strategy": "direct",
   "output_key_pattern": ""}}]"""

_L2_CODE_PROMPT = """\
You are writing a PYTHON PIPELINE (tier L2: a scaffold around the unchanged \
model) that repairs the failures described below.  Design any
pipeline you want — the only constraint is that the model itself is unchanged.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

EXECUTION CONTRACT:
- "{cases_file}" in the current directory: {{"cases": [{{"id": str, "prompt": str}}]}}
- A function  model_generate(case_id, prompt=None, image_ops=None) -> str  is
  ALREADY DEFINED in your namespace (do NOT import or redefine it).  It runs
  the ORIGINAL model on that case: optional prompt override, optional image
  transforms applied to the case's image first.  image_ops MUST be a list of
  {{"tool": "<name>", "params": {{...}}}} dicts using ONLY these tools
  (anything else is rejected with an error):
{catalog}{attend_hint}
- You may call the model SEVERAL times per case (budget ~6 calls/case) and
  branch on its outputs — e.g. ask where the finding could be, zoom there,
  re-ask; describe first, then decide; vote over variants.
- The LAST line of stdout MUST be exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- Emit an entry for EVERY case.  The "output" is scored externally against the
  original question, so it must answer that question faithfully (e.g. contain
  a clear yes/no for yes/no questions).
- Standard library + numpy only.  No network, no file writes.  Keep it under
  ~80 lines.

Return ONLY the Python code{fences_hint}."""

_REPAIR_PROMPT_BODY = """\
Your previously written repair pipeline FAILED TO EXECUTE.

ERROR:
{error}

YOUR PREVIOUS CODE:
```python
{code}
```

Fix the code.  Follow the execution contract EXACTLY:
- the ONLY model access is the predefined model_generate(case_id, prompt=None, \
image_ops=None){attend_clause} — do not import or redefine it;
- image_ops must be a list of {{"tool": "<name>", "params": {{...}}}} dicts \
using ONLY these tools:
{catalog}
- read "{cases_file}", emit an entry for EVERY case, and end stdout with \
exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- standard library + numpy only; no network, no file writes; under ~80 lines.
"""

_L3_PROMPT = """\
You are configuring WHITE-BOX intervention primitives (tier L3: the model's \
internals) against the failures below.  The primitives are pre-audited host \
code — you choose which to run and with what parameters.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

AVAILABLE PRIMITIVES:
{catalog}

Propose up to {k} configurations.  Reply with ONLY a JSON array:
[{{"primitive": "<name from the list>", "params": {{...}}}}]"""

_PAPER_METHOD_PROMPT = """\
You are selecting which PAPER-METHOD repair(s), if any, apply to the \
failure(s) below.  Each candidate is a specific, pre-implemented \
intervention that targets ONE named failure mechanism — read what mechanism \
each one actually targets, then select it ONLY when the verified hypotheses \
describe that same mechanism, not merely because it is technically able to \
run on this model and task (that eligibility has already been checked for \
you; your only job is judging whether the mechanism matches).

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

ELIGIBLE CANDIDATES (already filtered to what this model/task can run):
{catalog}

Propose up to {k} candidates whose targeted mechanism is genuinely \
consistent with the hypotheses above.  A hypothesis naming a DIFFERENT \
mechanism (e.g. a flat knowledge gap, a positional/answer-choice bias, or a \
hallucination cause when the candidate targets something else entirely) is \
NOT a match, even if the candidate would run without error.  If none of the \
candidates address the diagnosed mechanism, reply with an empty array — do \
not select one just because it is available.  Reply with ONLY a JSON array:
[{{"name": "<name from the list>", "rationale": "<one sentence: how the hypothesis's mechanism matches this candidate's>"}}]"""

_L4_PROMPT = """\
You are writing a PARAMETER-SPACE repair recipe (tier L4: fine-tuning) for \
the failures below.  The recipe is RECORDED for a human decision — it will \
not be executed automatically.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

Reply with ONLY a JSON object:
{{"dataset_recipe": "<how to build training data that generalises the failure
   mechanism — never just the observed failing cases>",
  "method": "lora|sft", "target": "vision_encoder|llm|projector|full",
  "eval_protocol": "<held-out repair effect + regression battery>",
  "rationale": "<why parameter-space change is the minimum effective tier>"}}"""
