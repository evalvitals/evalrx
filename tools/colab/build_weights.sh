#!/usr/bin/env bash
# Pack the notebooks' model weights into one .tar, for runtimes that cannot
# read them from their public source (gs://gemma-data for TPU, the Hugging Face
# Hub for GPU):
#
#   bash tools/colab/build_weights.sh tpu|gpu [out_dir]     # default out_dir: dist/
#
# The result, evalrx-colab-weights-<accel>-gemma-4-e2b.tar, unpacks to one
# directory that `evalrx run --model-path` takes: for TPU the Orbax checkpoint
# with tokenizer_gemma4.model inside it (18 GB), for GPU the Hugging Face
# snapshot of google/gemma-4-E2B-it. SHA256SUMS inside covers every file, and
# <name>.tar.sha256 next to it the tar, which launch.ipynb checks. It does not
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
NAME="evalrx-colab-weights-$ACCEL-gemma-4-e2b"
log() { echo "[evalrx-weights] $*"; }

# One pass: each file streams from its source into the tar, hashed on the way;
# nothing is staged on disk (Colab runtime disks can be slow).
if [ "$ACCEL" = gpu ]; then
  # The Hugging Face snapshot is staged, next to the output, then packed.
  STAGE="$(mktemp -d "$OUT_DIR/.weights.XXXXXX")"
  trap 'rm -rf "$STAGE"' EXIT
  log "downloading google/gemma-4-E2B-it"
  python3 - "$STAGE" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("google/gemma-4-E2B-it", local_dir=sys.argv[1])
PY
  rm -rf "$STAGE/.cache"
else
  STAGE=""
  # The checkpoint and tokenizer the jax_local backend reads by default (evalrx/specs.py).
  log "packing gs://gemma-data/checkpoints/gemma4-e2b-it and its tokenizer"
fi

python3 - "$OUT_DIR/$NAME.tar" "$NAME" "$STAGE" <<'PY'
import hashlib, io, json, os, sys, tarfile, time, urllib.parse, urllib.request
out, name, stage = sys.argv[1:]
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
    prefix = "checkpoints/gemma4-e2b-it/"
    for o in gcs_objects("gemma-data", prefix):
        if not o["name"].endswith(("_$folder$", "/")):  # "_$folder$": GCS console placeholders
            files.append((o["name"][len(prefix):], int(o["size"]), gcs("gemma-data", o["name"])))
    (tok,) = gcs_objects("gemma-data", "tokenizers/tokenizer_gemma4.model")
    files.append(("tokenizer_gemma4.model", int(tok["size"]), gcs("gemma-data", tok["name"])))
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
