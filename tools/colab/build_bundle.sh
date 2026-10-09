#!/usr/bin/env bash
# Build an offline Colab bundle: one .tar that bootstrap.sh installs with no
# network access at all (no GitHub, PyPI, PyTorch index or agy server).
#
#   bash tools/colab/build_bundle.sh gpu|tpu [out_dir]     # default out_dir: dist/
#
# Run it on any Linux x86_64 machine with internet, at a committed revision
# (the bundle contains `git archive HEAD`, not uncommitted edits). The result,
# evalrx-colab-<accel>-<commit>.tar, unpacks to evalrx/ = the source tree plus
#   .bundle/  uv, a relocatable Python 3.12, every locked wheel, agy, the frozen
#             notebook datasets (data/), and SHA256SUMS over all of it.
# Host it wherever the notebook can read: a GCS bucket, Google Drive, an
# internal file share. The notebook takes its location in EVALRX_BUNDLE.
#
# Before writing the tar, the bundle is installed offline into a scratch
# environment and used to freeze the datasets: a bundle that is missing a
# wheel fails here, not in Colab.
#
# Variables: EVALRX_BUNDLE_DATASETS (default: the notebooks' two datasets),
# EVALRX_SKIP_AGY=1, and the same index overrides as bootstrap.sh.
set -euo pipefail

ACCEL="${1:-}"
case "$ACCEL" in gpu|tpu) ;; *) echo "usage: build_bundle.sh gpu|tpu [out_dir]" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$(mkdir -p "${2:-$REPO_DIR/dist}" && cd "${2:-$REPO_DIR/dist}" && pwd)"
DATASETS="${EVALRX_BUNDLE_DATASETS:-bbh_word_sorting cruxeval_output}"
PY_VERSION=3.12
PYPI="${EVALRX_PYPI_INDEX:-https://pypi.org/simple}"
if [ "$ACCEL" = gpu ]; then TORCH_INDEX="${EVALRX_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
else TORCH_INDEX="${EVALRX_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"; fi
log() { echo "[evalrx-bundle] $*"; }

cd "$REPO_DIR"
COMMIT="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "WARNING: uncommitted changes are NOT in the bundle (it packs $COMMIT)"
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/evalrx"
B="$SRC/.bundle"
mkdir -p "$SRC" "$B/wheels" "$WORK/tool"
git archive HEAD | tar -x -C "$SRC"
echo "$COMMIT" > "$B/COMMIT"
echo "$ACCEL" > "$B/ACCEL"

# Wheels for the Colab target (Linux x86_64, CPython 3.12), whatever this host runs.
log "downloading wheels for tools/colab/lock/$ACCEL.txt"
python3 -m pip download -q --no-deps --only-binary=:all: --dest "$B/wheels" \
  --python-version "$PY_VERSION" --implementation cp --abi cp312 --abi abi3 --abi none \
  --platform manylinux_2_31_x86_64 --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform any \
  --index-url "$PYPI" --extra-index-url "$TORCH_INDEX" \
  -r "tools/colab/lock/$ACCEL.txt" setuptools wheel uv

# uv's static binary comes out of its wheel; the wheel itself is not needed.
uv_whl="$(ls "$B"/wheels/uv-*.whl)"
python3 - "$uv_whl" "$B/uv" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    name = next(n for n in z.namelist() if n.endswith("/scripts/uv"))
    open(sys.argv[2], "wb").write(z.read(name))
PY
chmod 0755 "$B/uv"
rm -f "$uv_whl"

# A relocatable python-build-standalone install, the same build uv uses online.
log "packing Python $PY_VERSION"
UV_PYTHON_INSTALL_DIR="$WORK/pythons" "$B/uv" python install -q --no-bin "$PY_VERSION"
py_root="$(dirname "$(dirname "$(UV_PYTHON_INSTALL_DIR="$WORK/pythons" "$B/uv" python find --managed-python "$PY_VERSION")")")"
tar -czf "$B/python.tar.gz" -C "$(dirname "$py_root")" "$(basename "$py_root")"

if [ "${EVALRX_SKIP_AGY:-0}" != 1 ]; then
  log "fetching agy"
  python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
    "${EVALRX_AGY_INSTALLER:-https://antigravity.google/cli/install.sh}" "$WORK/agy_install.sh"
  HOME="$WORK/home" bash "$WORK/agy_install.sh" --dir "$WORK/tool" >/dev/null
  install -m 0755 "$WORK/tool/agy" "$B/agy"
fi

(cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

# Self-test: offline install from the bundle, then freeze the datasets with it.
log "self-test: offline install"
HOME="$WORK/home" EVALRX_ENV="$WORK/env" UV_CACHE_DIR="$WORK/uv-cache" bash "$SRC/tools/colab/bootstrap.sh" "$ACCEL"
for ds in $DATASETS; do
  log "freezing dataset $ds"
  (cd "$WORK" && HOME="$WORK/home" "$WORK/env/bin/evalrx" run --modality llm --model gemma-4-e2b --dataset "$ds" \
     --download-limit 128 --seed 0 --download-only --data-dir "$B/data" >/dev/null)
done
(cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

out="$OUT_DIR/evalrx-colab-$ACCEL-${COMMIT:0:12}.tar"
tar -cf "$out" -C "$WORK" evalrx
log "wrote $out ($(du -h "$out" | cut -f1))"
log "sha256 $(sha256sum "$out" | cut -d' ' -f1)"
