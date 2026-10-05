"""Replay one frozen ChartQA repair on a TPU, with no judge or repair search.

The reference file contains the candidate chosen on EXPLORE and the CONFIRM
case IDs. This replay is a reproducibility check, not a new independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))


def materialize_data(reference, directory):
    """Recreate the frozen manifest and verify the image bytes before inference."""
    from PIL import Image

    rows = reference['manifest']
    directory.mkdir(parents=True, exist_ok=True)
    missing = [r for r in rows if not (directory / r['image']).is_file()]
    if missing:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        source = reference['dataset_source']
        table = pq.read_table(hf_hub_download(
            source['repo'], source['parquet'], repo_type='dataset', revision=source['revision']))
        for row in missing:
            item = table.slice(row['source_index'], 1).to_pylist()[0]
            if item['query'] + ' Answer with only the short answer.' != row['prompt']:
                raise ValueError(f"Source question changed: {row['id']}")
            destination = directory / row['image']
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB').save(destination)
    for row in rows:
        digest = hashlib.sha256((directory / row['image']).read_bytes()).hexdigest()
        if digest != reference['image_sha256'][row['id']]:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='Completed export containing a frozen candidate')
    p.add_argument('--data-dir', type=Path, default=Path(__file__).with_name('data') / 'chartqa')
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('outputs') / 'replay.json')
    p.add_argument('--tpu-url', help='Optional private JAX worker; otherwise load locally on TPU')
    p.add_argument('--limit', type=int, default=0, help='0 = all frozen CONFIRM cases; >0 is only a quick subset check')
    args = p.parse_args(argv)
    reference = json.loads(args.reference.read_text())
    if reference.get('status') != 'completed':
        raise ValueError('Replay requires a completed reference run')
    frozen = reference['candidate']
    if frozen is None:
        raise ValueError('The reference run did not select a repair; there is no candidate to replay')
    if frozen['kind'] not in ('template', 'spec'):
        raise ValueError('This replay only supports declarative L1/L2 candidates')
    manifest = materialize_data(reference, args.data_dir)

    from _common import tasks

    from evalrx.core.case import CaseBatch, Label
    from evalrx.eval_agent.stages.fix_agent import FixAgent, FixCandidate, FixOutcome
    from evalrx.eval_agent.stages.fix_tiers import parse_tier
    from evalrx.models import RuntimeConfig, compose
    from evalrx.specs import get_spec

    cases, _ = tasks.build_cases(tasks.get('chartqa'), manifest)
    by_id = {c.id: c for c in cases}
    ids = reference['confirm_ids']
    if args.limit:
        if args.limit < 0:
            p.error('--limit must be nonnegative')
        ids = ids[:args.limit]
    cases = [by_id[cid] for cid in ids]
    gen = reference['baseline_generation_kwargs']
    runtime = RuntimeConfig(device='tpu', dtype='bfloat16', apply_chat_template=True,
                            max_new_tokens=gen['max_new_tokens'], engine_kwargs={'text_only': False})
    if args.tpu_url:
        from run_chartqa_repair import remote_model
        model = remote_model(args.tpu_url.rstrip('/'), get_spec(reference['spec']), runtime)
    else:
        model = compose(reference['spec'], 'jax_local', runtime)
    model.load()
    if args.tpu_url:
        hardware = model.hardware
    else:
        import jax
        if jax.default_backend() != 'tpu':
            raise RuntimeError('This example requires actual TPU execution')
        hardware = {'device': jax.default_backend(), 'devices': [str(d) for d in jax.devices()]}
    result = {'status': 'running', 'kind': 'frozen_candidate_replay', 'hardware': hardware,
              'reference': str(args.reference), 'n': len(cases), 'baseline': [], 'fix': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

    save()
    try:
        labeled = []
        for index, case in enumerate(cases, 1):
            answer = model.generate(case.inputs, **gen)
            passed = tasks.score_case(case, answer)
            labeled.append(replace(case, observed=answer, label=Label.PASS if passed else Label.FAIL))
            result['baseline'].append({'id': case.id, 'expected': case.expected, 'observed': answer,
                                       'correct': passed})
            print(f'baseline {index}/{len(cases)} {case.id}: {answer!r} correct={passed}', flush=True)
            save()
        candidate = FixCandidate(tier=parse_tier(frozen['tier']), name=frozen['name'],
                                 kind=frozen['kind'], payload=frozen['payload'], source='frozen_reference')
        agent = FixAgent(judge=None, max_tier=frozen['tier'], allow_codegen=False,
                         score_fn=tasks.score_case, baseline_generation_kwargs=gen)
        validation = agent.validate_candidate(model, CaseBatch(labeled), candidate)
        result['fix'] = FixOutcome(max_tier=candidate.tier, attempted=[validation],
                                   fixed=validation.fixed).to_dict()
        complete = not validation.exec_error and validation.n_pairs == len(cases)
        result['status'] = 'completed' if complete else 'failed'
        result['predictions'] = []
        for row in result['baseline']:
            output = validation.outputs.get(row['id'])
            result['predictions'].append({**row, 'repaired': output,
                                          'repaired_correct': tasks.score_case(by_id[row['id']], output)
                                          if output is not None else None})
            print(f"repair {row['id']}: {output!r}", flush=True)
        print(f'CONFIRM n={validation.n_pairs}: baseline={validation.n_baseline_correct}, '
              f'repaired={validation.n_candidate_correct}, fixed={validation.n_fixed}, '
              f'broken={validation.n_broken}, verdict={validation.verdict}', flush=True)
        print(f'Output: {args.output}', flush=True)
        save()
        return int(result['status'] == 'failed')
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
