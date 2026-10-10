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

WORK="$(mktemp -d "$OUT_DIR/.weights.XXXXXX")"   # next to the output: it is as large as the tar
trap 'rm -rf "$WORK"' EXIT
DEST="$WORK/$NAME"
mkdir -p "$DEST"

if [ "$ACCEL" = tpu ]; then
  # The checkpoint and tokenizer the jax_local backend reads by default (evalrx/specs.py).
  log "downloading gs://gemma-data/checkpoints/gemma4-e2b-it"
  python3 - "$DEST" <<'PY'
import concurrent.futures, json, os, sys, urllib.parse, urllib.request
BUCKET, PREFIX, dest = "gemma-data", "checkpoints/gemma4-e2b-it/", sys.argv[1]
def objects(prefix):
    token = ""
    while True:
        q = urllib.parse.urlencode({"prefix": prefix, "pageToken": token, "fields": "items(name,size),nextPageToken"})
        page = json.load(urllib.request.urlopen(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{q}"))
        yield from page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return
def fetch(job):
    name, target, _ = job
    os.makedirs(os.path.dirname(target), exist_ok=True)
    urllib.request.urlretrieve(f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}", target)
    return os.path.getsize(target)
# "*_$folder$" objects are GCS console folder placeholders, not checkpoint files.
jobs = [(o["name"], os.path.join(dest, o["name"][len(PREFIX):]), int(o["size"]))
        for o in objects(PREFIX) if not o["name"].endswith(("_$folder$", "/"))]
jobs.append(("tokenizers/tokenizer_gemma4.model", os.path.join(dest, "tokenizer_gemma4.model"), None))
with concurrent.futures.ThreadPoolExecutor(16) as pool:
    sizes = list(pool.map(fetch, jobs))
bad = [j[0] for j, n in zip(jobs, sizes) if j[2] is not None and j[2] != n]
if bad:
    sys.exit(f"size mismatch: {bad[:5]}")
print(f"[evalrx-weights] {len(jobs)} files, {sum(sizes) / 1e9:.1f} GB")
PY
else
  REPO_ID=google/gemma-4-E2B-it
  log "downloading $REPO_ID"
  python3 - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
  rm -rf "$DEST/.cache"
fi

(cd "$DEST" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
out="$OUT_DIR/$NAME.tar"
tar -cf "$out" -C "$WORK" "$NAME"
(cd "$OUT_DIR" && sha256sum "$NAME.tar" > "$NAME.tar.sha256")
log "wrote $out ($(du -h "$out" | cut -f1))"
log "sha256 $(cut -d' ' -f1 "$out.sha256")"
