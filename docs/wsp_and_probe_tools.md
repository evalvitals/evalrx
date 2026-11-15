# Weight Edits and Probe Tools

Two additions that work together:

- **Probe tools** (`evalrx.probing`): every registered analyzer described as a
  tool — what it answers, what it needs, what arguments it takes, and whether
  it can run on the model and data in hand. The agentic loop can list them and
  call one by name.
- **Weight-space programming** (`evalrx.wsp`): build, apply, control and
  inspect direct weight edits such as `w' = w - alpha * u (u^T w)`, audit an
  edit against controls (`evalrx.analyzers.weights`), and try one as a
  training-free repair at fix tier L4.

## Probe tools

### Describe what can run

```python
from evalrx.probing import ProbeToolbox

box = ProbeToolbox(model, cases)
report = box.describe()
report["attention_knockout"]
# {"family": "causal", "access": "white_box", "cost": "heavy", "intervenes": True,
#  "question": "Which internal route carries the effect, measured by intervening on it?",
#  "parameters": {...JSON schema...},
#  "available": False,
#  "missing": ["needs a local white-box model that provides: attention, logits"],
#  "codes": ["capability"], ...}
```

Every probe is listed. An unavailable one says why, with all reasons rather
than the first, so "cannot run here" is never read as "ran and found nothing".
No forward pass is made. The reasons and their codes:

| Code | Meaning |
|---|---|
| `capability` | The model does not provide a required capability. |
| `modality` | The probe applies to modalities the model does not declare. |
| `data_slot` | The batch fills none of the modality slots the probe reads. |
| `trajectories` | The probe reads agent trajectories; the batch has none. |
| `data_precondition` | The cases lack labels or metadata the probe reads. |
| `injection` | A constructor argument must come from the host (a judge model, a weight edit). |
| `not_implemented` | Registered placeholder. |

The description is read from the analyzer class: family from its
`analyzers/<package>` location, arguments from its constructor, signals from
`signal_docs`. An analyzer may set `question`, `probe_cost`, `probe_family` or
`probe_intervenes` to override what is derived.

### Route a question to probes

```python
route = box.route(failure_modes=["visual_neglect"], max_probes=4)
route.names          # ranked: probes mapped to the failure mode first
route.unavailable    # requested probes that cannot run, with reasons
print(route.to_prompt())
```

Ranking is `StrategyProbe`'s (slot-composed priority plus failure-mode boosts),
restricted to probes that can run. Filters: `families`, `access`
(`"black_box"` / `"white_box"`), `allow_interventions`, `only_matching`.

### Call a probe by name

```python
result = box.call("logit_lens", {"top_k": 5})     # arguments checked against the schema
tools = box.tools()                               # evalrx.core.tool.Tool objects
tools[0].to_openai_schema()
```

A call with an unknown argument, a wrong type, or an unavailable probe raises
`ProbeCallError` before any model work, with the accepted arguments in the
message. Arguments that cannot be written as JSON are never part of a call;
pass a constructed analyzer through `overrides={name: analyzer}`.

### In the agentic loop

`AgenticDiagnoseLoop` gains one tool and extends another:

| Tool | Arguments | Effect |
|---|---|---|
| `list_probes` | `failure_modes`, `families`, `access`, `allow_interventions`, `only_matching`, `max_probes` | Shows the ranked runnable probes with arguments and reasons. Runs no model. Defaults to the failure modes of the current hypotheses. |
| `run_probe` | none | As before: the probe agent selects the analyzers. |
| `run_probe` | `name` + `params`, or `probes: [{name, params}]` | Runs exactly the named probes. Results are added to the evidence gathered so far. |

A named call runs what was named and nothing else: `max_analyzers` does not cap
it and no substitute is run for a probe that turns out inapplicable. A rejected
call does no model work. When the held-out confirmation re-runs the same probes
on the confirm split, it replays the arguments of the named call.

## Weight edits

### Build and apply

```python
from evalrx.wsp import TensorEdit, WeightEdit, applied, edit_structure

edit = WeightEdit([
    TensorEdit.rank_one("layers.15.mlp.down_proj.weight", u, alpha=6.0, cols=columns),
])
edit_structure(model, edit)        # size, location, rank, the alpha the change follows

with applied(model, edit) as report:
    answer = model.generate(case.inputs)
# the original weights are back, also if the block raised
```

A `TensorEdit` holds one tensor's change in the form it was written in: the
rank-one rule, low-rank factors, a dense delta, or a sparse overwrite.
`alpha = 1` removes the component along `u`, `alpha = 2` reflects it, larger
values reflect and amplify.

`applied` computes every edited tensor before writing any, so a bad name or
shape leaves the model untouched. `verify_base=True` requires the live tensors
to match the hashes recorded in the edit, and `max_relative_change` refuses an
edit above a size limit.

Tensor names match a parameter exactly, or by the suffix starting at the
indexed container, so an edit written for
`model.language_model.layers.15.mlp.down_proj.weight` applies to a module that
names it `model.layers.15.mlp.down_proj.weight`. The layer index is never
dropped, and a name that fits two parameters is an error.

`WeightEdit.save(dir)` / `WeightEdit.load(dir)` persist an edit.
`WeightEdit.from_astra(dir)` reads the `edit.npz` + `manifest.json` layout
published with *Sparks of Weight-Space Programming*.

### Derive an edit from failing and passing cases

```python
from evalrx.wsp import contrast_edit

edit = contrast_edit(model, fail_cases, pass_cases, n_layers=2, n_columns=64, alpha=1.0)
```

The direction at a layer is the difference between the mean MLP output on
failing and on passing cases; the neurons are the columns that push hardest
along it on failing cases. No gradient step is taken. This proposes an edit; it
does not show that the edit helps. Activations are read at the last prompt
position, so a failure that only develops during generation is not visible to
it.

For a failure that only exists during generation, contrast positions instead of
cases. `continuation_contrast_edit` appends each case's recorded output to its
prompt and contrasts positions inside a repeated stretch with positions outside
one:

```python
from evalrx.wsp import continuation_contrast_edit

edit = continuation_contrast_edit(model, looping_cases, repeat_ngram=32, n_columns=64, alpha=6.0)
```

`collect_continuation_stats` returns the per-layer group means on their own, and
`rank_columns` the direction with each neuron's push, projection and activation
difference, for comparing ranking signals.

### Compare and re-express edits

```python
from evalrx.wsp import edit_agreement, fit_rank_one

rule = fit_rank_one(model, published)      # u, columns and alpha that reproduce the edit
rule.metadata["fit"]                       # alpha and the unexplained share, per tensor
edit_agreement(model, mine, published)     # shared tensors, direction cosine, neuron overlap
```

An edit stored as overwritten entries cannot be scaled; its rank-one fit can,
which gives a dose curve over `alpha`. Check `rank_one_residual` before treating
the fit as the edit.

### Audit an edit

```python
from evalrx.analyzers.weights import WeightEditAudit, WeightEditStructure

WeightEditStructure(edit=edit).run(model, cases).findings
audit = WeightEditAudit(edit=edit, target="gold", n_seeds=3).run(model, cases).findings
```

`weight_edit_audit` measures the edit on the next-token distribution at the end
of the prompt, beside three alternatives on the same cases:

| Finding | Reading |
|---|---|
| `specificity_random_direction` | Output movement of the edit over that of an equally large change along a random direction. Near 1: the direction is not what matters. |
| `specificity_random_columns` | The same ratio against the same rule on other neurons. Near 1: the choice of neurons is not what matters. |
| `steer_recovery` | Share of the edit's per-case margin change reproduced by a constant activation shift with the weights untouched. Near 1: the edit acts like a fixed bias. |
| `mean_delta_edit_pass`, `frac_flips_pass` | Side effects on cases the model already got right. |

The controls are random draws; with few draws or few cases a ratio is an
estimate, not a test. Both analyzers are registered, so they appear in the
probe catalog under the `weight_edit` family once the host passes a constructed
instance:

```python
box = ProbeToolbox(model, cases, overrides={"weight_edit_audit": WeightEditAudit(edit=edit)})
box.call("weight_edit_audit", {"target": "gold", "n_seeds": 5})
```

### Audit an edit along a generation

Some edits act only during generation. `weight_edit_continuation_audit` appends
each case's recorded output (`case.observed`) to its prompt and scores every
continuation position on the token that actually came next:

```python
from evalrx.analyzers.weights import WeightEditContinuationAudit

findings = WeightEditContinuationAudit(
    edit=edit, groups="repetition", repeat_ngram=32, n_seeds=2,
).run(model, cases).findings
```

A prediction counts as in-loop when the `repeat_ngram` tokens before the
predicted token already occurred earlier in the sequence, and as productive
otherwise. Per group the findings give the change in negative log-likelihood of
the recorded token (`mean_dnll_<state>_<group>`), the KL divergence from the
unedited distribution, `specificity_<control>_<group>` and
`steer_recovery_<group>`; `dnll_edit_loop_minus_productive` is positive when the
edit makes looped text less likely than ordinary text.

This is teacher-forced on text the unedited model wrote. It measures what each
state does to the probability of that text, not what an edited model would
write. Positions within one generation are not independent, so compare the
`per_case` rows before drawing a conclusion from the pooled means. The model
handle must tokenise the prompt exactly as it was when the text was generated.

`tools/check_astra_edit_qwen35.py` and `tools/audit_astra_edit_qwen35.py` run
the structure check and this audit for the published p50c edit on Qwen3.5-0.8B;
`tools/recover_astra_edit_qwen35.py` derives an edit from the loop contrast and
compares it with the published one.

### As a repair (fix tier L4)

```python
agent = FixAgent(
    judge=judge, max_tier="L4", score_fn=score,
    finetune_pool=diagnosis_cases,          # labelled, disjoint from what is validated
    weight_edit={"alpha": 1.0, "n_columns": 64, "n_layers": 2},
)
outcome = agent.propose_and_validate(model, cases, hypotheses)
```

`weight_edit` is opt-in (`True` for the defaults, or a dict of
`WeightEditSpec` fields). The candidate `rank_one_weight_edit`:

- builds its edit from `finetune_pool` with `contrast_edit`, or loads one from
  `edit_dir`; a pool that shares cases with the validated batch is refused;
- generates the batch with the edit written in, then restores the weights;
- is validated by the same paired test as every other tier, breaks included;
- also generates the batch under a matched random-direction control and records
  both pass counts in `candidate.payload["audit"]`, with the edit's structure.

Without a pool (and without `edit_dir`) the candidate is recorded with the
reason it did not run. The judge never writes the edit or any hook code.

## What has and has not been checked

- The edit arithmetic, the restore contract, the structure read-out and the
  controls are tested on small tensors with known answers.
- The contrast heuristic, the audit and the fix path are tested on a hand-built
  model with one planted rank-one defect (`tests/toy_wsp.py`): the derived edit
  finds the planted layer, neuron and direction, repairs held-out cases, breaks
  none, and a matched random direction does not repair them.
- Mechanics (hooks, name resolution, restoration) are tested on tiny
  random-weight Qwen2.5-VL and Llama modules.
- The published-layout loader reproduces a re-implementation of the published
  rebuild rule bit for bit on synthetic bfloat16 tensors, and reads the
  published edit files. It has not been run against the original base
  checkpoint.
- On Qwen3.5-0.8B with the published p50c edit, the base and rebuilt tensor
  hashes match the manifest and the structure read-out gives 64 columns per
  layer, rank one, alpha = 6.000 at layers 15 and 19.
- The continuation audit of p50c on Qwen3.5-0.8B (12 unfinished GSM8K
  generations of the unedited model, 48,099 positions, teacher-forced) gives a
  change in next-token NLL of +0.079 at loop positions and +0.047 at productive
  positions; a matched random direction gives +0.011 / +0.010, the same rule on
  other neurons -0.000 / +0.000, and a constant activation shift of the edit's
  mean size +0.029 / +0.125. EditLens reports +0.093 / +0.053 for the edit on 6
  texts. The edit is specific and acts more in loops, and it is not free on
  ordinary text: the most likely token changes at 4.9% of productive positions.
- No generation-level result on a full-size model is recorded here. Whether a
  contrast-derived edit repairs real failures is an open empirical question.

## References

- Takehi, Teoh, Zhang & Khattab (2026). *Sparks of Weight-Space Programming.*
  <https://rikiyat.github.io/blog/weight-space-programming/>
- Zhang, Shang & Li (2026). *EditLens: What Did Astra's Edit Actually Do?*
  <https://xxlya.github.io/editlens/>
- VLMProbe: <https://github.com/ubc-tea/vlm-probe>
