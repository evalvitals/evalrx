"""Private, loopback-only JAX worker for running the repair controller elsewhere.

The normal tutorial runs directly on the TPU. This optional worker lets an
already-authenticated local judge use a Colab TPU over an SSH port forward.
Only generate/logprobs are exposed; no shell execution or file access API.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=18657)
    p.add_argument('--log-dir', required=True)
    args = p.parse_args()
    from evalrx.models import RuntimeConfig, compose
    from evalrx.models.backends.jax.backend import configure_jax_runtime
    configure_jax_runtime('tpu')
    import jax
    from PIL import Image

    from evalrx.core.case import Inputs
    if jax.default_backend() != 'tpu':
        raise RuntimeError('A real TPU backend is required')
    model = compose('gemma-4-e2b-it', 'jax_local', RuntimeConfig(
        device='tpu', dtype='bfloat16', apply_chat_template=True, max_new_tokens=64,
        engine_kwargs={'text_only': False}))
    model.load()
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    health = {'model': 'gemma-4-e2b-it', 'backend': 'jax_local', 'device': jax.default_backend(),
              'devices': [{'name': str(d), 'kind': d.device_kind} for d in jax.devices()]}
    (root / 'hardware.json').write_text(json.dumps(health, indent=2) + '\n')

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404, health if self.path == '/health' else {'error': 'unknown route'})

        def do_POST(self):
            started = time.monotonic()
            record = {'started_at_unix': time.time(), 'operation': self.path}
            try:
                if self.path not in ('/generate', '/logprobs'):
                    raise ValueError('unknown operation')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32 * 1024 * 1024:
                    raise ValueError('request size must be between 1 byte and 32 MiB')
                payload = json.loads(self.rfile.read(size))
                image = None
                if payload.get('image_png'):
                    raw = base64.b64decode(payload['image_png'], validate=True)
                    record['image_sha256'] = hashlib.sha256(raw).hexdigest()
                    image = Image.open(io.BytesIO(raw)).convert('RGB')
                prompt = str(payload['prompt'])
                kwargs = payload.get('kwargs', {})
                inputs = Inputs(prompt=prompt, image=image)
                record.update(prompt=prompt, kwargs=kwargs)
                if payload.get('inspect_encoding'):
                    from evalrx.models.backends.jax.adapters.gemma import OUT_BUCKETS, bucket_length
                    encoded = model.adapter.encode(inputs, chat_template=True)
                    budget = int(kwargs.get('max_tokens', kwargs.get('max_new_tokens', 64)))
                    padded = bucket_length(len(encoded.ids), model.adapter.buckets)
                    output_buffer = bucket_length(budget, OUT_BUCKETS)
                    record['allocation'] = {
                        'input_tokens': len(encoded.ids), 'padded_input_tokens': padded,
                        'requested_new_tokens': budget, 'output_buffer_tokens': output_buffer,
                        'old_cache_tokens': bucket_length(padded + output_buffer + 1, model.adapter.buckets),
                        'new_cache_tokens': bucket_length(padded + budget + 1, model.adapter.buckets),
                    }
                if self.path == '/generate':
                    result = {'text': model.generate(inputs, **kwargs)}
                else:
                    result = {'tokens': [asdict(t) for t in model.logprobs(inputs, **kwargs)]}
                record.update(status='passed', result=result)
                self.reply(200, result)
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                self.reply(500, {'error': record['error']})
            finally:
                record['seconds'] = round(time.monotonic() - started, 3)
                record['tpu_memory'] = jax.local_devices()[0].memory_stats()
                with (root / 'calls.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps({k:record[k] for k in ('operation','status','seconds')}), flush=True)

        def log_message(self, *args):
            pass

    print('READY', json.dumps(health), flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
