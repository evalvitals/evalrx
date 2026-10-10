#!/usr/bin/env bash
# Pack the installed EvalRX environment into a Colab image: one .tar that
# launch.ipynb restores by unpacking it, with no install step.
#
#   bash tools/colab/build_image.sh gpu|tpu [out_dir]     # default out_dir: /content/evalrx-image-out
#
# Run it on a Colab runtime with the target accelerator, after bootstrap.sh has
# built /content/evalrx-env from the source in /content/evalrx (setup.ipynb does
# all three steps). The image holds both directories at those fixed paths: the
# environment's scripts and its editable EvalRX install refer to them, so it
# must be restored to /content. Inside:
#   evalrx/          the source tree, with the frozen notebook datasets in .bundle/data
#   evalrx-env/      Python 3.12, every locked package, agy; .image = manifest (JSON)
# Next to the tar, <name>.sha256 holds its checksum; launch.ipynb checks it while
# unpacking. Freezing the datasets needs internet access; nothing else does.
set -euo pipefail

ACCEL="${1:-}"
case "$ACCEL" in gpu|tpu) ;; *) echo "usage: build_image.sh gpu|tpu [out_dir]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${EVALRX_ENV:-/content/evalrx-env}"
OUT_DIR="$(mkdir -p "${2:-/content/evalrx-image-out}" && cd "${2:-/content/evalrx-image-out}" && pwd)"
DATASETS="${EVALRX_BUNDLE_DATASETS:-bbh_word_sorting cruxeval_output}"
log() { echo "[evalrx-image] $*"; }
die() { echo "[evalrx-image] ERROR: $*" >&2; exit 1; }

[ "$REPO_DIR" = /content/evalrx ] && [ "$ENV_DIR" = /content/evalrx-env ] \
  || die "the image restores to /content/evalrx and /content/evalrx-env; build from there (got $REPO_DIR, $ENV_DIR)"
[ -f "$ENV_DIR/.stamp" ] && [ "$(cut -d' ' -f1 "$ENV_DIR/.stamp")" = "$ACCEL" ] \
  || die "no $ACCEL environment in $ENV_DIR: run tools/colab/bootstrap.sh $ACCEL first"
for var in $(compgen -e | grep -E '^(UV_|PIP_|PYTHONPATH$|PYTHONHOME$)' || true); do unset "$var"; done

COMMIT="$(cut -d' ' -f3 "$ENV_DIR/.stamp")"
mkdir -p "$REPO_DIR/.bundle"
echo "$COMMIT" > "$REPO_DIR/.bundle/COMMIT"
echo "$ACCEL" > "$REPO_DIR/.bundle/ACCEL"

for ds in $DATASETS; do
  if [ -d "$REPO_DIR/.bundle/data/$ds" ]; then log "dataset $ds already frozen"; continue; fi
  log "freezing dataset $ds"
  (cd "$OUT_DIR" && "$ENV_DIR/bin/evalrx" run --modality llm --model gemma-4-e2b --dataset "$ds" \
     --download-limit 128 --seed 0 --download-only --data-dir "$REPO_DIR/.bundle/data" >/dev/null)
done

# Smoke test the environment as launch.ipynb will use it, then record what it holds.
"$ENV_DIR/bin/evalrx" --help >/dev/null
[ -x "$ENV_DIR/bin/agy" ] || die "agy missing from $ENV_DIR/bin"
"$ENV_DIR/bin/python" - "$ACCEL" "$COMMIT" "$ENV_DIR/.image" <<'PY'
import datetime, importlib.metadata as md, json, os, platform, sys
accel, commit, out = sys.argv[1:]
names = ["evalrx", "torch", "transformers"] if accel == "gpu" else ["evalrx", "jax", "jaxlib", "libtpu", "gemma", "flax", "torch"]
for name in names:
    __import__(name)
manifest = {
    "accel": accel, "commit": commit, "python": platform.python_version(),
    "packages": {n: md.version(n) for n in names},
    "built_on": os.environ.get("COLAB_RELEASE_TAG", "non-Colab"),
    "accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE", ""),
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
open(out, "w").write(json.dumps(manifest, indent=1) + "\n")
print("[evalrx-image] manifest", json.dumps(manifest))
PY

name="evalrx-colab-$ACCEL-image-${COMMIT:0:12}.tar"
out="$OUT_DIR/$name"
log "writing $out"
# Left out: uv (only installs use it), the offline bundle's install inputs, git metadata.
# One pass: the checksum is computed while the tar is written.
tar -cf - -C /content \
  --exclude=evalrx-env/.uv --exclude=evalrx/.git --exclude=evalrx/.bundle/wheels \
  --exclude=evalrx/.bundle/python.tar.gz --exclude=evalrx/.bundle/uv --exclude=evalrx/.bundle/agy \
  --exclude=evalrx/.bundle/SHA256SUMS \
  evalrx evalrx-env | tee "$out" | sha256sum | sed "s| -\$| $name|" > "$out.sha256"
log "wrote $out ($(du -h "$out" | cut -f1))"
log "sha256 $(cut -d' ' -f1 "$out.sha256")"
