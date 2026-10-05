"""ChartQA + Gemma E2B on TPU: automatic L1/L2 repair with a frozen CONFIRM gate.

Runs the benchmark's own baseline, M1, structured M2/M3, M4 and FixAgent.
L1/L2 repairs change prompts or inference pipelines, not weights. Optional free-form exploration,
code generation and surgery are disabled; no repair template is supplied.

python examples/colab/run_chartqa_repair.py --judge-provider claude --judge-model claude-sonnet-4-6

Use --tpu-url http://127.0.0.1:18657 with tools/colab/serve_jax.py to keep the
controller and its authenticated judge on another machine over an SSH tunnel.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/benchmark'))


def remote_model(url, spec, runtime):
    from evalrx.core.capability import Capability
    from evalrx.core.model import Model, TokenLogprob

    class RemoteJaxModel(Model):
        capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
        modalities = frozenset({'text', 'image'})

        def __init__(self):
            self.spec, self.runtime = spec, runtime

        def load(self):
            with urllib.request.urlopen(url + '/health', timeout=30) as response:
                self.hardware = json.load(response)
            if self.hardware['device'] != 'tpu' or self.hardware['model'] != spec.key:
                raise RuntimeError(f'Unexpected remote model: {self.hardware}')
            return self

        def call(self, operation, inputs, kwargs):
            from PIL import Image
            prompt = getattr(inputs, 'prompt', str(inputs))
            image = getattr(inputs, 'image', None)
            payload = {'prompt': prompt, 'kwargs': kwargs}
            if image is not None:
                im = image if isinstance(image, Image.Image) else Image.open(image)
                buf = io.BytesIO()
                im.convert('RGB').save(buf, format='PNG')
                payload['image_png'] = base64.b64encode(buf.getvalue()).decode()
            req = urllib.request.Request(url + '/' + operation, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=900) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors='replace')
                raise RuntimeError(f'TPU worker HTTP {exc.code}: {detail}') from exc

        def generate(self, inputs, **kwargs):
            return self.call('generate', inputs, kwargs)['text']

        def logprobs(self, inputs, **kwargs):
            return [TokenLogprob(**t) for t in self.call('logprobs', inputs, kwargs)['tokens']]

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError('This optional transport exposes only generation and logprobs')

    return RemoteJaxModel()


def main(argv=None):
    from _common import runner, tasks
    from _common.models import resolve
    from _common.run import build_parser

    p = build_parser()
    p.add_argument('--tpu-url', help='Private JAX worker URL (otherwise run natively on the TPU)')
    p.set_defaults(modality='vlm', model='gemma-4-e2b', backend='jax_local', device='tpu',
                   dataset='chartqa', limit=128, download_limit=128, seed=5022,
                   data_dir=str(ROOT / 'examples/colab/data'), run_dir=str(ROOT / 'examples/colab/outputs'),
                   run_tag='tpu_repair', fix_tier='L2', allow_codegen=False, explore=False,
                   m2_codegen=False, skip_m5=True, fix_validation_cases=64,
                   temperature=0.0, max_new_tokens=64)
    args = p.parse_args(argv)
    task = tasks.get(args.dataset)
    resolved = resolve(args.model, args.modality, args.backend)
    original_load = runner.load_model
    def load_tpu(resolved, args, task):
        output = runner.run_dir_for(args, task)
        output.mkdir(parents=True, exist_ok=True)
        if args.tpu_url:
            from evalrx.models import RuntimeConfig
            from evalrx.specs import get_spec
            spec = get_spec(resolved.spec_key)
            gen = runner.generation_settings(task, args)
            model = remote_model(args.tpu_url.rstrip('/'), spec, RuntimeConfig(
                device='tpu', max_new_tokens=gen['max_new_tokens'], apply_chat_template=True))
            model.load()
            hardware = model.hardware
        else:
            import importlib.metadata
            import platform

            model, gen, spec = original_load(resolved, args, task)
            import jax
            if jax.default_backend() != 'tpu':
                raise RuntimeError('This example requires actual TPU execution')
            hardware = {'model': spec.key, 'backend': 'jax_local', 'device': jax.default_backend(),
                        'devices': [{'name': str(d), 'kind': d.device_kind} for d in jax.devices()]}
            packages = {}
            for name in ('evalrx', 'gemma', 'jax', 'jaxlib', 'libtpu', 'torch', 'flax', 'kauldron', 'numpy', 'pillow'):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    pass
            (output / 'environment.json').write_text(json.dumps(
                {'python': platform.python_version(), 'packages': packages}, indent=2) + '\n')
        (output / 'tpu_hardware.json').write_text(json.dumps(hardware, indent=2) + '\n')
        return model, gen, spec
    runner.load_model = load_tpu
    try:
        return runner.run(args, task, resolved)
    finally:
        runner.load_model = original_load


if __name__ == '__main__':
    raise SystemExit(main())
