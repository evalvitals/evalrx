# Default agentic Gemini benchmark validation

These are full frozen-manifest runs: BBH causal judgement 187, ChartQA 128, MMAU 128.
Target model: Gemini 3.6 Flash. Judge/coder: Claude Opus 5, high effort. Concurrency: 1.
No diagnosis-mode or M1-M5 skip flags. This is a new agentic protocol, not a model-only historical fixed-pipeline comparison.

## Provenance

- `provenance.json`: commands, images, output locations, and the first failed BBH attempt.
- `source_sha256.json`: frozen source snapshot used by image/audio runs and initial BBH attempt.
- `source_sha256_llm_retry.json`: corrected runner snapshot for BBH retry.
- `runs/llm/`: initial failure (duplicate `floor_candidates` during FixAgent construction), preserved.
- `runs/llm-retry/`, `runs/vlm/`, `runs/alm/`: complete-chain attempts; inspect terminal state before claiming completion.
- `progress.json`: monitoring snapshot, not a final result.
- `recovered_exploration/`: original exploration workspaces recovered before temporary cleanup; these running snapshots predate the fix routing exploration through artifact publication and M3 context capture (commit 7285788).

## Interpretation

All baselines were regenerated: BBH retry 136/187, ChartQA 90/128, MMAU 89/128.
These counts are not repair gains. Final frozen-candidate confirmation and evidence gates determine repair success.
The actual API probability probe returned `Logprobs is not enabled for this model`; raw logits are unavailable.

After the three latest attempts terminate, `python examples/benchmark/smoke_checks/agentic-default-20261012/audit.py` verifies split isolation, confirmation scoring, source snapshots, stage records, and container exits, and writes `full_chain_audit.json`.

## Final audited results

All three latest full-chain attempts exited 0. Offline rescoring matched the recorded confirmation counts; EXPLORE and CONFIRM IDs were disjoint. Frozen source snapshots and the three original manifest hashes were unchanged.

| Dataset | Confirm correct before → after | Fixed / broken | e-value (threshold 20) | Verdict |
| --- | --- | --- | --- | --- |
| bbh_causal_judgement | 68/94 → 66/94 | 4 / 6 | 0.4433 | unsafe |
| chartqa | 45/64 → 57/64 | 13 / 1 | 78.0190 | fixed |
| mmau | 44/64 → 50/64 | 6 / 0 | 9.1429 | partial |

Selection evaluated 13 BBH, 10 ChartQA, and 15 MMAU candidates, followed by one frozen candidate per confirmation set. MMAU recovered from a real M3 Claude timeout through the new bounded retry path.

The final source regression suite passed 426 tests. The later exploration publication/context fix was regression-tested; these frozen live runs were not restarted with that later patch.
