#!/usr/bin/env python3
"""Check every prerequisite BEFORE the expensive part starts.

The failure this exists to prevent: the chain spends 40 minutes generating a
batch on the GPU and only then discovers that the `claude` CLI is missing, so
M1 cannot select analyzers and the whole run is wasted. Everything checkable in
under a second is checked here instead.

    python preflight.py                          # check everything
    python preflight.py --model qwen3.5-9b --dataset supergpqa_law
    python preflight.py --skip-net                # offline: skip the HF probe

Exit 0 = ready. Exit 1 = something REQUIRED is missing (each line says what to
install). Warnings never fail the run on their own.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))

#: GPU memory each model needs at the defaults run_all.sh serves with
#: (--max-model-len 32768 --gpu-memory-utilization 0.92). Below this, lower
#: --max-model-len rather than hoping.
VRAM_GB = {"qwen3.5-2b": 12, "qwen3.5-4b": 20, "qwen3.5-9b": 40}
WEIGHTS_GB = {"qwen3.5-2b": 5, "qwen3.5-4b": 9, "qwen3.5-9b": 21}

_fail: list = []
_warn: list = []


def ok(msg: str) -> None:
    print(f"  \033[32mOK\033[0m    {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"  \033[31mFAIL\033[0m  {msg}\n        -> {fix}")
    _fail.append(msg)


def warn(msg: str, fix: str = "") -> None:
    print(f"  \033[33mWARN\033[0m  {msg}" + (f"\n        -> {fix}" if fix else ""))
    _warn.append(msg)


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        ok(f"python {v.major}.{v.minor}.{v.micro} (>=3.10)")
    else:
        bad(f"python {v.major}.{v.minor} is too old",
            "evalvitals requires >=3.10; make a new venv")


def check_imports() -> None:
    # (module, what needs it, pip extra)
    required = [
        ("yaml", "config parsing", "pyyaml"),
        ("numpy", "everything", "numpy"),
        ("requests", "dataset fetch + endpoint calls", "requests"),
        ("evalvitals", "the pipeline itself", 'pip install -e ".[stats,viz,dashboard]"'),
        ("statsmodels", "M2 statistics", 'pip install -e ".[stats]"'),
        ("sklearn", "M2 statistics", 'pip install -e ".[stats]"'),
    ]
    optional = [
        ("matplotlib", "M2 charts", 'pip install -e ".[viz]"'),
        ("streamlit", "the dashboard command", 'pip install -e ".[dashboard]"'),
    ]
    for mod, why, fix in required:
        if importlib.util.find_spec(mod) is not None:
            ok(f"import {mod} ({why})")
        else:
            bad(f"missing module {mod} — needed for {why}", fix)
    for mod, why, fix in optional:
        if importlib.util.find_spec(mod) is not None:
            ok(f"import {mod} ({why})")
        else:
            warn(f"missing module {mod} — {why} will not work", fix)


def check_claude() -> None:
    exe = shutil.which("claude")
    if not exe:
        bad("`claude` CLI not on PATH — M1/M2/M3/M5 judges cannot run",
            "install Claude Code and authenticate: https://claude.com/claude-code")
        return
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            ok(f"claude CLI {out.stdout.strip().splitlines()[0]}")
        else:
            bad("`claude --version` failed — probably not authenticated",
                "run `claude` once interactively to log in")
            return
    except (subprocess.TimeoutExpired, OSError) as exc:
        bad(f"`claude --version` did not respond ({type(exc).__name__})",
            "check the CLI install")
        return

    # A bad --model or --effort would otherwise surface only at M1, i.e. AFTER
    # the batch has been generated on the GPU. One real call settles it.
    try:
        import yaml
        cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    except Exception:
        warn("cannot read config.yaml; skipped the judge round-trip")
        return
    model, effort = str(cfg.get("judge_model", "")), str(cfg.get("judge_effort", ""))
    cmd = [exe]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    cmd += ["-p", "Reply with exactly the word OK"]
    try:
        # high/xhigh effort thinks for a while even on a trivial prompt
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                             cwd="/tmp")
    except subprocess.TimeoutExpired:
        warn(f"judge probe (model={model} effort={effort}) took >5 min",
             "works, but M1-M5 will be slow; consider a lower effort")
        return
    except OSError as exc:
        bad(f"judge probe failed to launch ({type(exc).__name__})", "check the CLI")
        return
    if out.returncode == 0 and out.stdout.strip():
        ok(f"judge round-trip: model={model} effort={effort}")
    else:
        detail = (out.stderr or out.stdout).strip().splitlines()
        bad(f"judge probe failed for model={model!r} effort={effort!r}: "
            f"{detail[0][:120] if detail else 'empty reply'}",
            "fix judge_model / judge_effort in config.yaml "
            "(effort must be low|medium|high|xhigh|max)")


def check_vllm() -> None:
    exe = os.environ.get("VLLM_BIN") or shutil.which("vllm")
    if exe and Path(exe).exists():
        ok(f"vllm binary: {exe}")
    else:
        bad("`vllm` not found — cannot serve the model under test",
            "pip install vllm==0.27.1 in a SEPARATE venv, then export VLLM_BIN=...")


def check_gpu(model: str) -> None:
    if not shutil.which("nvidia-smi"):
        bad("nvidia-smi not found — no GPU visible", "this pipeline needs a CUDA GPU")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        bad("nvidia-smi did not respond", "check the driver")
        return
    free = []
    for line in out.stdout.strip().splitlines():
        idx, name, total, used = (p.strip() for p in line.split(","))
        avail = (int(total) - int(used)) / 1024
        if avail > 1.0:
            free.append((idx, name, avail))
    need = VRAM_GB.get(model, 40)
    if not free:
        bad("every GPU is busy", "wait, or pass GPU=<idx> to override")
        return
    best = max(free, key=lambda t: t[2])
    if best[2] >= need:
        ok(f"GPU {best[0]} ({best[1]}) has {best[2]:.0f} GB free, {model} needs ~{need}")
    else:
        warn(f"largest free GPU has {best[2]:.0f} GB, {model} wants ~{need} GB at the "
             f"default --max-model-len 32768",
             "lower --max-model-len / --gpu-memory-utilization, or use a smaller model")


def check_disk(model: str) -> None:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    probe = cache if cache.exists() else Path.home()
    try:
        free_gb = shutil.disk_usage(probe).free / 1024 ** 3
    except OSError:
        warn(f"cannot stat {probe}")
        return
    need = WEIGHTS_GB.get(model, 21)
    hit = list(cache.glob(f"hub/models--Qwen--{model.replace('qwen', 'Qwen').replace('.', '.')}*"))
    if hit:
        ok(f"weights already cached under {cache}")
    elif free_gb >= need:
        ok(f"{free_gb:.0f} GB free at {probe}; {model} weights are ~{need} GB "
           f"(will download on first serve)")
    else:
        bad(f"only {free_gb:.0f} GB free at {probe}, {model} needs ~{need} GB",
            "free space or set HF_HOME to a larger volume")


def check_network(skip: bool) -> None:
    if skip:
        warn("skipped the HuggingFace probe (--skip-net)")
        return
    try:
        import requests
        # /valid was retired and now 404s -- probe a real query instead, or the
        # health check fails on a perfectly working network
        r = requests.get("https://datasets-server.huggingface.co/is-valid",
                         params={"dataset": "cais/mmlu"}, timeout=20)
        if r.status_code == 200:
            ok("datasets-server reachable")
        else:
            bad(f"datasets-server returned HTTP {r.status_code}",
                "datasets are fetched at runtime; a proxy or mirror is required")
    except Exception as exc:
        bad(f"cannot reach datasets-server ({type(exc).__name__})",
            "set HTTPS_PROXY, or pre-download the datasets")


def check_catalog(dataset: str) -> None:
    try:
        import datasets as CAT
    except Exception as exc:
        bad(f"cannot import the dataset catalog ({type(exc).__name__}: {exc})",
            "run from inside examples/llm_benchmark")
        return
    try:
        entry = CAT.get(dataset)
        acq = CAT.acquisition(dataset)
        ok(f"dataset {dataset!r} resolves ({entry.items} items, "
           f"9B reference {entry.accuracy_9b:.3f})")
        ok(f"  -> {acq['dataset']} config={acq['config']} split={acq['split']}"
           + (f" where={acq['where']}" if acq["where"] else ""))
    except SystemExit as exc:
        bad(str(exc), f"pick one of: {', '.join(sorted(CAT.BY_NAME))}")
    except StopIteration:
        bad(f"{dataset!r} is in the catalog but has no band_locate spec",
            "the two have drifted; run `python datasets.py`")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--dataset", default="supergpqa_law")
    ap.add_argument("--skip-net", action="store_true")
    args = ap.parse_args()

    print(f"preflight: model={args.model} dataset={args.dataset}")
    print(f"  package root: {PKG_ROOT}")
    print()
    check_python()
    check_imports()
    check_claude()
    check_vllm()
    check_gpu(args.model)
    check_disk(args.model)
    check_network(args.skip_net)
    check_catalog(args.dataset)

    print()
    if _fail:
        print(f"\033[31m{len(_fail)} REQUIRED check(s) failed\033[0m — fix them before "
              f"running; the chain would waste GPU time and then die.")
        return 1
    print(f"\033[32mready\033[0m" + (f" ({len(_warn)} warning(s))" if _warn else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
