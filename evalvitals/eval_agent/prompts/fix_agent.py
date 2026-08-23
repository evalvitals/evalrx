"""Prompt templates for fix candidate generation and repair."""

_L1_PROMPT = """\
You are designing PROMPT-LEVEL fixes (tier L1: the input space only) for a \
model failure. The model may be text-only or vision-language.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}
{context}
EXAMPLE CASES — read the model's own outputs: what it does when it fails, what
it does when it succeeds, and the answer format the scorer expects:
{examples}

Propose up to {k} prompt rewrite strategies that could repair these failures
WITHOUT changing the model or adding pipeline steps.  Each strategy is a
template applied to every case prompt; it MUST contain the literal placeholder
{{prompt}}.  Keep the final-answer format the scorer expects recoverable; do
not ask the model to suppress its reasoning if the task needs it.


"what_it_does" is shown to a reader who has never heard of this model, this
benchmark, or this field -- write it for a bright high-school student.  One
sentence, present tense, plain words, saying what CHANGES for the model.  Do
not name the tier, do not use metric or method jargon, and do not restate the
snake_case name in English.
Good:  "Asks the model to describe what it hears before it answers."
Bad:   "L1 audio-evidence-first prompt scaffold with deferred answering."

Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>", "what_it_does": "<one plain sentence>",
   "prompt_template": "<template with {{prompt}}>"}}]"""

_L2_PROMPT = """\
You are designing SCAFFOLD-LEVEL fixes (tier L2: a pipeline around the \
unchanged model) for a model failure. The model may be text-only or
vision-language.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}
{context}
EXAMPLE CASES — read the model's own outputs: what it does when it fails, what
it does when it succeeds, and the answer format the scorer expects:
{examples}

IMAGE TOOLS (only meaningful when the case has an image):
{catalog}

Propose up to {k} pipelines. Each may chain image tools, rewrite the prompt
(template MUST contain {{prompt}}), sample the model n_samples times (1..5;
applies to EVERY strategy — a multi-call strategy is repeated end-to-end and
its final answers vote on the extracted final answer), select one reviewed
multi-call strategy (direct, least_to_most, self_refine,
chain_of_verification), and set bounded decoding controls in
generation_kwargs (temperature, top_p, stop, max_tokens). max_tokens can only
RAISE the baseline budget — a value below it is raised to the baseline, so
never try to shorten the model's chain of thought; temperature 0 makes
n_samples>1 pointless. For structured answer tasks, output_key_pattern may be
a regex with one capture group used for answer-only voting; it must not
contain gold answers.

"what_it_does" is shown to a reader who has never heard of this model, this
benchmark, or this field -- write it for a bright high-school student.  One
sentence, present tense, plain words, saying what CHANGES for the model.  Do
not name the tier, do not use metric or method jargon, and do not restate the
snake_case name in English.
Good:  "Asks the model to describe what it hears before it answers."
Bad:   "L1 audio-evidence-first prompt scaffold with deferred answering."

Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>", "what_it_does": "<one plain sentence>",
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
{context}
EXAMPLE CASES — read the model's own outputs: what it does when it fails, what
it does when it succeeds, and the answer format the scorer expects:
{examples}

EXECUTION CONTRACT:
- "{cases_file}" in the current directory: {{"cases": [{{"id": str, "prompt": str,
  "baseline_output": str|null}}]}} — baseline_output is the model's ORIGINAL
  recorded answer to that prompt: the DIRECT BASELINE.  It may be wrong; it is
  NOT a label and carries no correctness information.  Use it directly as the
  baseline answer (compare against it, vote with it, ask the model to
  double-check it) — you do NOT need to call model_generate(case_id) to
  obtain it; a plain model_generate(case_id) is answered from this record.
- A function  model_generate(case_id, prompt=None, image_ops=None,
  generation_kwargs=None) -> str  is ALREADY DEFINED in your namespace (do NOT
  import or redefine it).  It runs the ORIGINAL model on that case: optional
  prompt override, optional image transforms applied to the case's image
  first, optional bounded decoding controls (temperature, top_p, stop,
  max_tokens — max_tokens can only RAISE the baseline budget; a lower value is
  raised to it).  image_ops MUST be a list of
  {{"tool": "<name>", "params": {{...}}}} dicts using ONLY these tools
  (anything else is rejected with an error):
{catalog}{attend_hint}
- You may call the model SEVERAL times per case, but keep it to at most
  4 calls per case that hit the model (genuinely independent enhanced/reasoned
  passes: a different prompt and/or image_ops each); the host enforces this
  per-case cap.  A plain model_generate(case_id) (no prompt override, no
  image_ops, no generation_kwargs) is answered from the recorded
  baseline_output, costs nothing and does not count.  Branch on outputs
  within that budget — e.g. describe first, then decide; vote over
  independent variants.
- model_generate is thread-safe and concurrent calls are serviced in
  parallel: fan out over CASES with concurrent.futures.ThreadPoolExecutor
  (max_workers=8) — a serial loop over every case x several calls is slow and
  risks the wall-clock limit.
- SELECTION RULE (enforced by the host, without labels): for each case the
  host anchors on baseline_output (or on your plain direct call when a case
  has no record).  Your final "output" may differ from that anchor only when
  at least {min_support} of your enhanced calls with DISTINCT prompt/image_ops
  returned the same answer; otherwise the host reverts that case to the
  anchor.  Return the model's own reply text (or the tagged answer you asked
  it to emit, e.g. "FINAL: <answer>" — the host strips such tags when
  matching); never rewrite, reformat or compute the answer yourself.
  Pattern the host accepts:
      base = case["baseline_output"] or ""
      votes = [model_generate(cid, prompt=p1), model_generate(cid, prompt=p2, image_ops=ops)]
      final = the answer that >= {min_support} votes agree on if it differs from base, else base
{selection_guidance}
- The LAST line of stdout MUST be exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- Emit an entry for EVERY case.  The "output" is scored externally against the
  original question, so it must answer that question faithfully (e.g. contain
  a clear yes/no for yes/no questions).
- Standard library + numpy only.  No network, no file writes.  Keep it under
  ~80 lines.
- REPAIR THE MODEL, NOT THE TASK.  Your code is re-run a second time with every
  model_generate() call answered by the model's ORIGINAL recorded answer.  Any
  failing case your pipeline gets right in THAT run was solved by your code,
  not the model, and is excluded from the fix score — a pipeline that computes
  the answer itself (sorting, arithmetic, lookup, a hard-coded default) scores
  zero.  Verify, vote, re-ask, re-prompt, zoom: make the model get it right.

The FIRST line of your code must be a WHAT_IT_DOES comment, on ONE line, for a
reader who has never heard of this model, this benchmark, or this field: plain
words, present tense, saying what your pipeline makes the model do differently.
No jargon, no tier names, no restating the code.  For example:
  # WHAT_IT_DOES: Asks the model twice with different wording and keeps the answer both tries agree on.

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
image_ops=None, generation_kwargs=None){attend_clause} — do not import or redefine it;
- image_ops must be a list of {{"tool": "<name>", "params": {{...}}}} dicts \
using ONLY these tools:
{catalog}
- baseline_output in "{cases_file}" IS the direct baseline answer — use it as \
the baseline; a plain model_generate(case_id) is answered from that record and \
is free; at most 4 model-hitting calls per case;
- the host's selection guard reverts a case to its baseline unless at least \
{min_support} DISTINCT enhanced calls returned your final answer (answer tags \
such as "FINAL: <answer>" are stripped when matching);
{selection_guidance}
- read "{cases_file}", emit an entry for EVERY case, and end stdout with \
exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- standard library + numpy only; no network, no file writes; under ~80 lines.
- repair the MODEL, not the task: the code is re-run with every model_generate() \
call answered by the model's original recorded answer, and failing cases it still \
gets right then are excluded from the score.  If it timed out, make FEWER model \
calls per case — do not replace them with code that computes the answer itself.
- keep (or add) the one-line `# WHAT_IT_DOES: ...` first line describing, in \
plain words for a reader with no background, what your pipeline makes the model \
do differently.
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
