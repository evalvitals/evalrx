"""Prompt templates for the M5 experiment writer."""

from typing import Any

_BLUEPRINT_SYSTEM = """\
You are an expert ML debugging engineer. Plan the structure of a multi-file
Python diagnostic project that will test a specific hypothesis about a model's
behaviour.

Output a YAML specification with this structure:
```yaml
description: one-line summary
files:
  - name: main.py
    generation_order: 3
    description: entry point
    pseudocode: |
      1. Load model
      2. Run cases
      3. Compute metrics
      4. Print metrics in name: value format
      5. Print verdict: 1.0 or verdict: 0.0
    dependencies: [model_probe.py, utils.py]
  - name: utils.py
    generation_order: 1
    description: shared helpers
    pseudocode: |
      - helper functions
    dependencies: []
output_contract:
  metrics: [mean_consistency, n_cases]
  verdict: "1.0 = hypothesis SUPPORTED, 0.0 = REFUTED"
```

Rules:
- Generate 2–4 files maximum
- main.py must be last in generation_order
- main.py MUST have `if __name__ == "__main__":` block
- The last printed line of the entire run MUST be `verdict: 1.0` or `verdict: 0.0`
- Stay under {timeout_sec} seconds total runtime\
"""

_BLUEPRINT_USER = """\
## Hypothesis to test
Statement   : {statement}
Failure mode: {failure_mode}

## Model setup
```python
{import_expr}
model = {load_expr}
```

Available capabilities: {capabilities}

## Model access policy
{model_access_note}

## Failure cases (JSON)
```json
{cases_json_snippet}
```

Write the YAML blueprint now:\
"""

_GENERATE_FILE_SYSTEM = """\
You are an expert ML debugging engineer. Generate a single Python file
as part of a multi-file diagnostic experiment project.

Rules:
- Implement EXACTLY the pseudocode in the file spec
- Use only the evalrx APIs shown
- Output ONLY the Python code — no markdown prose before or after
- Code must be complete and immediately executable\
"""

_GENERATE_FILE_USER = """\
## File to generate: {file_name}

### File spec
```json
{file_spec}
```

### Blueprint (full project)
```yaml
{blueprint}
```

### Dependencies already generated
{dep_summaries}

{dep_code}

## Model access policy
{model_access_note}

## EvalRX API reference
```python
from evalrx.core.case import CaseBatch, FailureCase, Inputs
output: str = model.generate(case.inputs)
# logprobs (only if 'LOGPROBS' in capabilities)
lps = model.logprobs(case.inputs)
# internals (only if 'ATTENTION' or 'HIDDEN_STATES' in capabilities)
trace = model.forward(case.inputs, capture={{Capability.ATTENTION}})
```

Write the complete `{file_name}` now (Python code only):\
"""

_WRITE_SYSTEM = """\
You are an expert ML debugging engineer. Write a concise, self-contained Python
diagnostic script that gathers evidence for or against a specific hypothesis
about a model's behaviour.

RULES:
- Use ONLY the evalrx APIs shown in the user message.
- The script must be completely self-contained — no external files.
- Print ALL metrics in EXACTLY this format (one per line):
    metric_name: float_value
- Print a final verdict line as the LAST output:
    verdict: 1.0    # hypothesis is SUPPORTED by the evidence
    verdict: 0.0    # hypothesis is REFUTED
- Never print anything after the verdict line.
- Catch exceptions and print error metrics rather than crashing silently.
- Stay under {timeout_sec} seconds total runtime.
- The script MUST have `if __name__ == "__main__":` block.\
"""

_WRITE_USER = """\
## Hypothesis to test
Statement  : {statement}
Failure mode: {failure_mode}

## Model setup (copy this verbatim)
```python
{import_expr}
model = {load_expr}
```

Available capabilities: {capabilities}

## Model access policy
{model_access_note}

{blueprint_context}

## EvalRX API reference
```python
from evalrx.core.case import CaseBatch, FailureCase, Inputs
from evalrx.core.capability import Capability

# Text generation (always available)
output: str = model.generate(case.inputs)

# Token logprobs  (only if 'LOGPROBS' in capabilities)
lps = model.logprobs(case.inputs)   # list[TokenLogprob]; each has .token, .logprob

# Internals capture (only if 'ATTENTION' or 'HIDDEN_STATES' in capabilities)
trace = model.forward(case.inputs, capture={{Capability.ATTENTION}})
# trace.attentions     list[Tensor]  one per layer, shape [heads, seq, seq]
# trace.hidden_states  list[Tensor]  one per layer, shape [seq, hidden]
# trace.tokens         list[str]
```

## Failure cases (JSON — deserialize inside your script)
```json
{cases_json}
```

## Required output (EXACTLY this format)
```
metric_a: 0.72
metric_b: 3.0
verdict: 1.0
```

Write the complete Python script now:\
"""

_REPAIR_SYSTEM = """\
You are a Python debugging expert. The script below crashed.
Return the COMPLETE corrected script — no explanations, just the fixed code.
Preserve all logic; fix only the error.\
"""

_REPAIR_USER = """\
## Error
```
{error}
```

## Script to fix
```python
{code}
```\
"""

_REVIEW_SYSTEM = """\
You are a senior code reviewer. Review this diagnostic experiment code and
return a JSON object with:
  - "verdict": "APPROVE" or "REQUEST_CHANGES"
  - "score": integer 1-10
  - "critical_issues": list of strings (empty if APPROVE)

Focus on: correctness of hypothesis testing logic, metric computation,
verdict output format, and error handling.  Do NOT flag style issues.\
"""


def build_cli_prompt(
    hypothesis: Any,
    model_context: dict[str, Any],
    timeout_sec: int,
    n_images: int = 0,
) -> str:
    """Build the prompt used by external CLI coding agents."""
    caps = ", ".join(model_context.get("capabilities", [])) or "GENERATE"
    access_note = model_context.get("access_note", "")
    image_note = (
        f", and image_path (JPEG path for {n_images} case(s)).\n"
        "Load images and build inputs like this:\n"
        "```python\n"
        "from PIL import Image\n"
        "from evalrx.core.case import Inputs\n"
        "img = Image.open(case['image_path'])\n"
        "inputs = Inputs(prompt=case['prompt'], image=img)\n"
        "# Then: model.generate(inputs) or model.forward(inputs, capture={Capability.ATTENTION})\n"
        "```\n"
        "Do NOT pass raw dicts or lists as inputs — always use Inputs().\n\n"
        if n_images else ".\n\n"
    )
    return (
        "You are an ML debugging engineer. Write a self-contained Python "
        "diagnostic script that tests a specific hypothesis about a model.\n\n"
        "## Hypothesis\n"
        f"Statement   : {hypothesis.statement}\n"
        f"Failure mode: {hypothesis.predicted_failure_mode}\n\n"
        "## Model setup (copy verbatim)\n"
        "```python\n"
        f"{model_context.get('import_expr', 'import evalrx')}\n"
        f"model = {model_context.get('load_expr', '# model')}\n"
        "```\n\n"
        f"Available capabilities: {caps}\n\n"
        "## Model access policy\n"
        f"{access_note}\n\n"
        "## Input data\n"
        "Read `cases.json` from the current directory. "
        "Each record has: prompt, label (PASS/FAIL), id, metadata"
        + image_note
        + "## Required output (print to stdout, EXACTLY this format)\n"
        "```\n"
        "metric_a: 0.72\n"
        "metric_b: 3.0\n"
        "verdict: 1.0    # 1.0=SUPPORTED  0.0=REFUTED\n"
        "```\n\n"
        "## Rules\n"
        f"- Stay under {timeout_sec} seconds total runtime.\n"
        "- Save the script as `experiment.py` in the current directory.\n"
        "- Print all metrics as `name: float_value` lines.\n"
        "- The last printed line must be `verdict: 1.0` or `verdict: 0.0`.\n"
        "- Use only the evalrx APIs shown above.\n"
        "- Do NOT make external network calls or read files other than `cases.json`.\n"
    )
