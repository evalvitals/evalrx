"""jax_local smoke: does Gemma-4-E2B load and serve every read path on this device?

usage: python tools/colab/jax_smoke.py --device tpu|auto|cpu|cuda [--text-only] [--media]
       [--ckpt DIR] [--sharding auto|fsdp|single]
Each step prints `STEP <name> OK|FAIL <secs> <detail>`; a failed step does not
stop the later ones (except load).
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback


def mem(jax):
    import resource
    host = f"host_maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB "
    try:
        s = jax.devices()[0].memory_stats() or {}
        return host + f"hbm in_use={s.get('bytes_in_use', 0) / 1e9:.2f}GB peak={s.get('peak_bytes_in_use', 0) / 1e9:.2f}GB limit={s.get('bytes_limit', 0) / 1e9:.2f}GB"
    except Exception as exc:  # noqa: BLE001
        return f"memory_stats unavailable ({exc})"


def step(name, fn):
    t0 = time.monotonic()
    try:
        detail = fn()
        print(f"STEP {name} OK {time.monotonic() - t0:.1f}s {detail}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"STEP {name} FAIL {time.monotonic() - t0:.1f}s {type(exc).__name__}: {str(exc)[:600]}", flush=True)
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--media", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--sharding", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("absl").setLevel(logging.WARNING)

    from evalrx.core.capability import Capability
    from evalrx.core.case import Inputs
    from evalrx.models import RuntimeConfig, compose

    kw = {"text_only": bool(args.text_only)}
    if args.ckpt:
        kw["checkpoint"] = args.ckpt
    if args.sharding:
        kw["sharding"] = args.sharding
    rt = RuntimeConfig(device=args.device, dtype=args.dtype, max_new_tokens=24,
                       apply_chat_template=True, engine_kwargs=kw)
    m = compose("gemma-4-e2b-it", "jax_local", rt)
    print("MODEL", m, "caps", sorted(c.name for c in m.capabilities), "modalities", sorted(m.modalities), flush=True)

    import jax  # after compose: load() sets JAX_PLATFORMS first

    if not step("load", lambda: (m.load(), f"backend={jax.default_backend()} devices={jax.devices()} {mem(jax)}")[1]):
        print("SUMMARY load failed", mem(jax))
        return
    q = "What is the capital of France? Answer in one word."
    step("generate_greedy_cold", lambda: repr(m.generate(q)))
    step("generate_greedy_warm", lambda: repr(m.generate(q)))
    step("generate_sampled_x4", lambda: repr([m.generate("Write one sentence about anything you like.", temperature=1.0) for _ in range(4)]))

    def lp():
        out = m.logprobs(q, max_new_tokens=6, top_k=3)
        return repr([(t.token, round(t.logprob, 3)) for t in out])
    step("logprobs", lp)

    def fwd():
        caps = {Capability.LOGITS, Capability.HIDDEN_STATES}
        if Capability.ATTENTION in m.capabilities:
            caps.add(Capability.ATTENTION)
        tr = m.forward(q, capture=caps)
        att = None if tr.attentions is None else (len(tr.attentions), tuple(tr.attentions[0].shape), str(tr.attentions[0].dtype))
        return (f"seq={len(tr.token_ids)} logits={tuple(tr.logits.shape)}/{tr.logits.dtype} "
                f"hidden={len(tr.hidden_states)}x{tuple(tr.hidden_states[0].shape)}/{tr.hidden_states[0].dtype} "
                f"attn={att} provided={sorted(c.name for c in tr.provided)} {mem(jax)}")
    step("forward_capture", fwd)

    def lens():
        from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer
        r = LogitLensAnalyzer(top_k=3).run(m, "The capital of France is")
        return f"{type(r).__name__} findings={len(getattr(r, 'findings', []) or [])}"
    step("logit_lens_analyzer", lens)

    if args.media and not args.text_only:
        import numpy as np
        from PIL import Image

        img = Image.new("RGB", (850, 600), (200, 30, 30))
        iq = Inputs(prompt="What colour is this? Answer with one word.", image=img)

        def image():
            tr = m.forward(iq, capture={Capability.LOGITS})
            return f"grids={tr.token_type_map.grids} answer={m.generate(iq)!r}"
        step("image", image)
        wav = np.zeros(16000 * 2, dtype=np.float32)
        aq = Inputs(prompt="Is there speech in this clip? Answer yes or no.", audio=wav)

        def audio():
            tr = m.forward(aq, capture={Capability.LOGITS})
            return f"audio_tokens={int(tr.extras['audio_token_mask'].sum())} answer={m.generate(aq)!r}"
        step("audio", audio)
    print("SUMMARY done", mem(jax), flush=True)


if __name__ == "__main__":
    main()
