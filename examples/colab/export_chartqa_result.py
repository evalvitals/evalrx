"""Export a completed run's frozen repair, actual outputs and reproducibility data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))


def export(run_dir, manifest, output, dataset_revision, worker_log=None):
    # summary.json is only written after the runner finalizes all stages.
    summary = json.loads((run_dir / 'summary.json').read_text())
    baseline = json.loads((run_dir / 'baseline.json').read_text())
    log = json.loads((run_dir / 'logs/M5/log.json').read_text())
    outcome = log['fix'][-1]
    attempts = outcome['attempted']
    if len(attempts) > 1:
        raise ValueError('Expected at most one frozen candidate on CONFIRM')
    attempt = attempts[0] if attempts else None
    if attempt and attempt['name'] != outcome['selected_on_explore']:
        raise ValueError('CONFIRM candidate differs from the EXPLORE selection')
    rows = json.loads(manifest.read_text())
    baseline_by_id = {r['id']: r for r in baseline['cases']}
    predictions = []
    if attempt:
        for cid, answer in attempt['outputs'].items():
            before = baseline_by_id[cid]
            predictions.append({'id': cid, 'expected': before['expected'],
                                'baseline': before['observed'], 'repaired': answer,
                                'baseline_correct': before['label'] == 'pass',
                                'fixed': cid in attempt['fixed_cases'],
                                'broken': cid in attempt['broken_cases']})
    from _common import tasks

    cases, _ = tasks.build_cases(tasks.get('chartqa'), manifest)
    cases = {c.id: c for c in cases}
    for row in predictions:
        if tasks.score_case(cases[row['id']], row['baseline']) != row['baseline_correct']:
            raise ValueError(f"Baseline score changed for {row['id']}")
        row['repaired_correct'] = tasks.score_case(cases[row['id']], row['repaired'])
        fixed = not row['baseline_correct'] and row['repaired_correct']
        broken = row['baseline_correct'] and not row['repaired_correct']
        if row['fixed'] != fixed or row['broken'] != broken:
            raise ValueError(f"Paired case classification changed for {row['id']}")
    if attempt:
        counts = {
            'n_pairs': len(predictions),
            'n_baseline_correct': sum(r['baseline_correct'] for r in predictions),
            'n_candidate_correct': sum(r['repaired_correct'] for r in predictions),
            'n_fixed': sum(not r['baseline_correct'] and r['repaired_correct'] for r in predictions),
            'n_broken': sum(r['baseline_correct'] and not r['repaired_correct'] for r in predictions),
        }
        if any(attempt[key] != value for key, value in counts.items()):
            raise ValueError(f'Cannot export a simple paired replay: scored outputs disagree with {counts}')
    result = {
        'status': 'completed',
        'description': 'Actual TPU run; frozen candidate selected on EXPLORE, evaluated once on CONFIRM.',
        'spec': baseline['spec'], 'model': baseline['hf_repo'], 'dataset': baseline['dataset'],
        'checkpoint': ({'gemma-4-e2b-it': 'gs://gemma-data/checkpoints/gemma4-e2b-it'}
                       .get(baseline['spec'])),
        'hardware': json.loads((run_dir / 'tpu_hardware.json').read_text()),
        'environment': json.loads((run_dir / 'environment.json').read_text())
                       if (run_dir / 'environment.json').is_file() else None,
        'baseline_generation_kwargs': baseline['generation_kwargs'],
        'dataset_source': {'repo': 'HuggingFaceM4/ChartQA',
                           'parquet': 'data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet',
                           'revision': dataset_revision},
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'manifest': rows,
        'image_sha256': {r['id']: hashlib.sha256((manifest.parent / r['image']).read_bytes()).hexdigest()
                         for r in rows},
        'baseline_all': {k: baseline[k] for k in ('n', 'n_pass', 'n_fail', 'accuracy', 'seconds')},
        'baseline_outputs': baseline['cases'],
        'candidate': {k: attempt[k] for k in ('name', 'tier', 'kind', 'source', 'payload')} if attempt else None,
        'confirm_ids': [r['id'] for r in predictions],
        'confirm': {k: attempt[k] for k in (
            'n_pairs', 'n_baseline_correct', 'n_candidate_correct', 'n_fixed', 'n_broken',
            'effect', 'e_value', 'e_threshold', 'fixed', 'verdict', 'n_unstable', 'noise_model',
        )} if attempt else None,
        'predictions': predictions,
        'selection_attempted': outcome['selection_attempted'],
        'fixed': outcome['fixed'], 'recommendation': outcome.get('recommendation'),
        'run_summary': summary,
        'provenance': json.loads((run_dir / 'provenance.json').read_text())
                      if (run_dir / 'provenance.json').is_file() else None,
        'source_hash_scope': 'Current source at export time; provenance records launch-time core hashes.',
        'source_files_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT / 'examples/colab/run_chartqa_repair.py',
                                          ROOT / 'examples/colab/replay_chartqa_repair.py',
                                          ROOT / 'examples/benchmark/_common/scoring.py',
                                          ROOT / 'examples/benchmark/_common/tasks/chartqa.py',
                                          ROOT / 'tools/colab/serve_jax.py',
                                          ROOT / 'evalrx/models/backends/jax/adapters/gemma.py']},
    }
    if worker_log:
        raw_calls = worker_log.read_bytes()
        calls = [json.loads(line) for line in raw_calls.splitlines() if line.strip()]
        failures = [call for call in calls if call.get('status') != 'passed']
        archived_log = output.with_suffix('.worker_calls.jsonl.gz')
        archived_log.parent.mkdir(parents=True, exist_ok=True)
        archived_log.write_bytes(gzip.compress(raw_calls, mtime=0))
        result['worker_execution_audit'] = {
            'scope': 'Worker lifetime through export, including the initial 300-token regression check.',
            'sha256': hashlib.sha256(raw_calls).hexdigest(),
            'raw_calls_gzip': archived_log.name,
            'calls': len(calls), 'failed_calls': len(failures),
            'failed_configurations': dict(Counter(json.dumps(call.get('kwargs', {}), sort_keys=True)
                                                  for call in failures)),
            'errors': dict(Counter(call.get('error', 'unknown') for call in failures)),
            'interpretation': 'Affected candidates have execution failures and missing scored pairs; '
                              'their scores are not complete 64-case accuracy comparisons.',
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('spec', 'dataset', 'baseline_all', 'confirm', 'fixed')}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-revision', required=True)
    p.add_argument('--worker-log', type=Path, help='Optional raw TPU worker log for execution-failure accounting')
    args = p.parse_args()
    export(args.run_dir, args.manifest, args.output, args.dataset_revision, args.worker_log)


if __name__ == '__main__':
    main()
