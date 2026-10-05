"""Validate Gemma E2B on JAX and write an optional JSON report.

python tools/colab/jax_smoke.py --device tpu --text-only --output text.json
python tools/colab/jax_smoke.py --device tpu --suite image --output image.json
python tools/colab/jax_smoke.py --device tpu --suite audio --output audio.json

Use separate processes for the suites to release compiled programs between them.
A failed check is recorded and makes the process exit nonzero.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import resource
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def memory(jax):
    # Linux ru_maxrss is KiB; accelerator counters are bytes.
    result = {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    result["devices"] = []
    for d in jax.local_devices():
        stats = d.memory_stats() or {}
        result["devices"].append({"device": str(d), **{
            k: int(stats[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in stats
        }})
    return result


class SmokeReport:
    def __init__(self, output=None):
        self.output = Path(output) if output else None
        self.data = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "steps": []}

    def save(self):
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.output.with_suffix(self.output.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2) + "\n")
            tmp.replace(self.output)

    def step(self, name, fn):
        start = time.monotonic()
        record = {"name": name}
        try:
            record.update(status="passed", detail=fn())
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        record["seconds"] = round(time.monotonic() - start, 3)
        self.data["steps"].append(record)
        self.save()
        print("STEP", json.dumps(record), flush=True)
        return record["status"] == "passed"

    def finish(self):
        ok = bool(self.data["steps"]) and all(s["status"] == "passed" for s in self.data["steps"])
        self.data.update(status="passed" if ok else "failed", finished_at=datetime.now(timezone.utc).isoformat())
        self.save()
        print("SUMMARY", self.data["status"], flush=True)
        return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--suite", choices=("text", "image", "audio"), default="text")
    ap.add_argument("--media", action="store_true", help="Also check image and audio in this process")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--sharding", choices=("auto", "fsdp", "single"), default="auto")
    ap.add_argument("--output", help="JSON report, updated after each step")
    args = ap.parse_args(argv)
    if args.text_only and (args.media or args.suite != "text"):
        ap.error("--text-only cannot be combined with a media suite or --media")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("absl").setLevel(logging.WARNING)
    report = SmokeReport(args.output)
    report.data["config"] = vars(args)
    report.data["python"] = platform.python_version()
    report.data["versions"] = {}
    for name in ("evalrx", "gemma", "jax", "jaxlib", "libtpu", "flax", "kauldron", "torch"):
        try:
            report.data["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report.data["versions"][name] = None
    report.save()

    from evalrx.core.capability import Capability
    from evalrx.core.case import Inputs
    from evalrx.models import RuntimeConfig, compose
    from evalrx.models.backends.jax.backend import configure_jax_runtime

    # Configure before importing JAX, including in this smoke's own diagnostics.
    configure_jax_runtime(args.device)
    import jax
    import torch

    def hardware():
        backend = jax.default_backend()
        expected = {"cuda": "gpu", "gpu": "gpu", "tpu": "tpu", "cpu": "cpu"}.get(args.device.split(":")[0])
        assert expected is None or backend == expected, f"Requested {expected}, got {backend}"
        return {"backend": backend, "devices": [{"name": str(d), "kind": d.device_kind} for d in jax.devices()],
                "process_count": jax.process_count(), "memory": memory(jax)}

    if not report.step("hardware", hardware):
        return report.finish()
    kw = {"text_only": args.text_only, "sharding": args.sharding}
    if args.ckpt:
        kw["checkpoint"] = args.ckpt
    rt = RuntimeConfig(device=args.device, dtype=args.dtype, max_new_tokens=24,
                       apply_chat_template=True, engine_kwargs=kw)
    m = compose("gemma-4-e2b-it", "jax_local", rt)

    def load():
        m.load()
        return {"capabilities": sorted(c.name for c in m.capabilities), "modalities": sorted(m.modalities),
                "memory": memory(jax)}

    if not report.step("load", load):
        return report.finish()

    def answer(inputs):
        text = m.generate(inputs)
        assert text.strip(), "Empty generation"
        return {"answer": text, "memory": memory(jax)}

    if args.suite == "text":
        q = "What is the capital of France? Answer in one word."
        report.step("generate_greedy_cold", lambda: answer(q))
        report.step("generate_greedy_warm", lambda: answer(q))

        def sample():
            texts = [m.generate("Write one sentence about anything you like.", temperature=1.0) for _ in range(4)]
            assert all(t.strip() for t in texts), "Empty sampled generation"
            return {"answers": texts, "unique_answers": len(set(texts))}
        report.step("generate_sampled_x4", sample)

        def lp():
            import math
            out = m.logprobs(q, max_new_tokens=6, top_k=3)
            assert out and all(math.isfinite(t.logprob) and t.logprob <= 1e-5 for t in out)
            return {"tokens": [{"token": t.token, "logprob": t.logprob} for t in out]}
        report.step("logprobs", lp)

        def fwd():
            caps = {Capability.LOGITS, Capability.HIDDEN_STATES, Capability.ATTENTION}
            tr = m.forward(q, capture=caps)
            assert caps <= tr.provided
            assert len(tr.hidden_states) == 36 and len(tr.attentions) == 35
            assert tr.logits.shape == (len(tr.token_ids), 262144)
            assert all(torch.isfinite(t).all() for t in [tr.logits, *tr.hidden_states, *tr.attentions])
            attn_error = max(float((a.float().sum(-1) - 1).abs().max()) for a in tr.attentions)
            assert attn_error < 0.02, f"Attention rows do not sum to 1: {attn_error}"
            return {"seq": len(tr.token_ids), "logits": list(tr.logits.shape),
                    "hidden_layers": len(tr.hidden_states), "attention_layers": len(tr.attentions),
                    "attention_shape": list(tr.attentions[0].shape), "dtype": str(tr.logits.dtype),
                    "trace_device": str(tr.logits.device), "attention_row_sum_max_error": attn_error,
                    "memory": memory(jax)}
        report.step("forward_capture", fwd)

        def lens():
            from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer
            r = LogitLensAnalyzer(top_k=3).run(m, "The capital of France is")
            assert r.findings["n_layers"] == 36 and r.findings["n_cases_analyzed"] == 1
            return {"result_type": type(r).__name__, "findings": r.findings}
        report.step("logit_lens_analyzer", lens)

    if args.suite == "image" or args.media:
        from PIL import Image
        iq = Inputs(prompt="What colour is this? Answer with one word.", image=Image.new("RGB", (850, 600), (200, 30, 30)))

        def image():
            tr = m.forward(iq, capture={Capability.LOGITS})
            assert torch.isfinite(tr.logits).all()
            count = int(tr.extras["image_token_mask"].sum())
            assert count > 0 and tr.token_type_map.grids
            return {"image_tokens": count, "grids": tr.token_type_map.grids, **answer(iq)}
        report.step("image", image)

    if args.suite == "audio" or args.media:
        import numpy as np
        aq = Inputs(prompt="Is there speech in this clip? Answer yes or no.", audio=np.zeros(16000 * 2, dtype=np.float32))

        def audio():
            tr = m.forward(aq, capture={Capability.LOGITS})
            assert torch.isfinite(tr.logits).all()
            count = int(tr.extras["audio_token_mask"].sum())
            assert count > 0
            return {"audio_tokens": count, **answer(aq)}
        report.step("audio", audio)
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
