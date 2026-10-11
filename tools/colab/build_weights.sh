#!/usr/bin/env bash
# Pack a model's weights into one .tar, for runtimes that cannot read them from
# the source the model's spec registers (evalrx/specs.py: gs://gemma-data for
# TPU, the Hugging Face Hub for GPU). Runs normally read that source directly;
# this tar is only for runtimes without access to it.
#
#   bash tools/colab/build_weights.sh tpu|gpu [out_dir] [model]   # defaults: dist/, gemma-4-e2b
#
# The result, evalrx-colab-weights-<accel>-<model>.tar, unpacks to one
# directory that `evalrx run --model-path` takes: for TPU the spec's Orbax
# checkpoint with its tokenizer inside it, for GPU the spec's Hugging Face
# snapshot. SHA256SUMS inside covers every file, and
# <name>.tar.sha256 next to it the tar, which tpu/evalrx_tpu.ipynb checks. It does not
# depend on the EvalRX revision, so one weights tar serves every bundle. Host it
# next to the bundle and set EVALRX_WEIGHTS in the notebook's first cell.
#
# Needs internet access, and for GPU Python with huggingface_hub. TPU weights
# come from the public bucket over plain HTTPS (no gcloud needed).
set -euo pipefail

ACCEL="${1:-}"
case "$ACCEL" in gpu|tpu) ;; *) echo "usage: build_weights.sh gpu|tpu [out_dir]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$(mkdir -p "${2:-$REPO_DIR/dist}" && cd "${2:-$REPO_DIR/dist}" && pwd)"
MODEL="${3:-gemma-4-e2b}"
NAME="evalrx-colab-weights-$ACCEL-$MODEL"
log() { echo "[evalrx-weights] $*"; }

# The model's registered sources: the JAX checkpoint and tokenizer (TPU), the Hugging Face repo (GPU).
PY="${EVALRX_ENV:-/content/evalrx-env}/bin/python"; [ -x "$PY" ] || PY=python3
read -r HF_REPO CKPT TOKENIZER < <(PYTHONPATH="$REPO_DIR" "$PY" - "$MODEL" <<'PY'
import sys
from evalrx.benchmark.models import SIZES
from evalrx.specs import get_spec
spec = get_spec(SIZES[sys.argv[1]].specs["llm"])
print(spec.hf_repo, spec.jax.checkpoint if spec.jax else "-", spec.jax.tokenizer if spec.jax else "-")
PY
)

# One pass: each file streams from its source into the tar, hashed on the way;
# nothing is staged on disk (Colab runtime disks can be slow).
if [ "$ACCEL" = gpu ]; then
  # The Hugging Face snapshot is staged, next to the output, then packed.
  STAGE="$(mktemp -d "$OUT_DIR/.weights.XXXXXX")"
  trap 'rm -rf "$STAGE"' EXIT
  log "downloading $HF_REPO"
  python3 - "$STAGE" "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[2], local_dir=sys.argv[1])
PY
  rm -rf "$STAGE/.cache"
else
  STAGE=""
  case "$CKPT $TOKENIZER" in gs://*" "gs://*) ;; *) echo "[evalrx-weights] ERROR: $MODEL has no gs:// JAX checkpoint and tokenizer" >&2; exit 1 ;; esac
  log "packing $CKPT and $TOKENIZER"
fi

python3 - "$OUT_DIR/$NAME.tar" "$NAME" "$STAGE" "$CKPT" "$TOKENIZER" <<'PY'
import hashlib, io, json, os, sys, tarfile, time, urllib.parse, urllib.request
out, name, stage, ckpt, tokenizer = sys.argv[1:]
GCS = "https://storage.googleapis.com"

def gcs_objects(bucket, prefix):
    token = ""
    while True:
        q = urllib.parse.urlencode({"prefix": prefix, "pageToken": token, "fields": "items(name,size),nextPageToken"})
        page = json.load(urllib.request.urlopen(f"{GCS}/storage/v1/b/{bucket}/o?{q}"))
        yield from page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return

files = []  # (path inside the weights directory, size, opener)
if stage:
    for root, _, names in os.walk(stage):
        for n in sorted(names):
            path = os.path.join(root, n)
            files.append((os.path.relpath(path, stage), os.path.getsize(path), lambda path=path: open(path, "rb")))
else:
    def gcs(bucket, obj):
        return lambda: urllib.request.urlopen(f"{GCS}/{bucket}/{urllib.parse.quote(obj)}")
    bucket, _, prefix = ckpt[len("gs://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    for o in gcs_objects(bucket, prefix):
        if not o["name"].endswith(("_$folder$", "/")):  # "_$folder$": GCS console placeholders
            files.append((o["name"][len(prefix):], int(o["size"]), gcs(bucket, o["name"])))
    tok_bucket, _, tok_name = tokenizer[len("gs://"):].partition("/")
    (tok,) = gcs_objects(tok_bucket, tok_name)  # next to the checkpoint, where the adapter looks for it
    files.append((os.path.basename(tok_name), int(tok["size"]), gcs(tok_bucket, tok["name"])))
files.sort()

class Hashing(io.RawIOBase):
    """Pass reads (or writes) through, hashing the bytes."""
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()
    def readable(self): return True
    def writable(self): return True
    def read(self, n=-1):
        b = self.f.read(n); self.h.update(b); return b
    def write(self, b):
        self.h.update(b); return self.f.write(b)

sums, total, start = [], sum(f[1] for f in files), time.time()
with open(out, "wb") as raw:
    tar_out = Hashing(raw)
    with tarfile.open(fileobj=tar_out, mode="w|", format=tarfile.GNU_FORMAT) as tar:
        done = 0
        for rel, size, opener in files:
            info = tarfile.TarInfo(f"{name}/{rel}")
            info.size, info.mode, info.mtime = size, 0o644, int(start)
            with opener() as src:
                reader = Hashing(src)
                tar.addfile(info, reader)  # raises if the source ends early
            sums.append(f"{reader.h.hexdigest()}  ./{rel}\n")
            done += size
            print(f"[evalrx-weights] {done / 1e9:5.1f}/{total / 1e9:.1f} GB  {rel}", flush=True)
        data = "".join(sums).encode()
        info = tarfile.TarInfo(f"{name}/SHA256SUMS")
        info.size, info.mode, info.mtime = len(data), 0o644, int(start)
        tar.addfile(info, io.BytesIO(data))
digest = tar_out.h.hexdigest()
with open(out + ".sha256", "w") as f:
    f.write(f"{digest}  {os.path.basename(out)}\n")
print(f"[evalrx-weights] wrote {out} ({os.path.getsize(out) / 1e9:.1f} GB, {len(files)} files, {time.time() - start:.0f} s)")
print(f"[evalrx-weights] sha256 {digest}")
PY
