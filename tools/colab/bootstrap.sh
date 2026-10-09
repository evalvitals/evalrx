#!/usr/bin/env bash
# Build a self-contained EvalRX environment for a Colab notebook:
#
#   bash tools/colab/bootstrap.sh gpu|tpu
#
# The environment has its own Python 3.12 and the exact packages in
# tools/colab/lock/<accel>.txt, in $EVALRX_ENV (default /content/evalrx-env).
# It never imports or modifies the notebook kernel's Python, so a Colab image
# update (new Python, new preinstalled jax/torch) does not change it. Run every
# EvalRX command through $EVALRX_ENV/bin.
#
# Two sources, same result:
#   online   uv, Python, wheels and agy come from the network (overridable below)
#   offline  this checkout contains .bundle/ (made by build_bundle.sh): nothing
#            is downloaded. Use it where PyPI/GitHub are unreachable.
#
# Network overrides (online mode):
#   EVALRX_PYPI_INDEX     package index   (default: pip's configured index, else pypi.org)
#   EVALRX_TORCH_INDEX    torch wheels    (default: download.pytorch.org/whl/{cu129,cpu})
#   EVALRX_PYTHON_MIRROR  Python builds   (uv's UV_PYTHON_INSTALL_MIRROR; default GitHub)
#   EVALRX_AGY_INSTALLER  agy installer   (default antigravity.google/cli/install.sh)
#   EVALRX_SKIP_AGY=1     do not install the agy coding agent
set -euo pipefail

ACCEL="${1:-}"
case "$ACCEL" in gpu|tpu) ;; *) echo "usage: bootstrap.sh gpu|tpu" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="$REPO_DIR/tools/colab/lock/$ACCEL.txt"
BUNDLE="$REPO_DIR/.bundle"
if [ -z "${EVALRX_ENV:-}" ]; then
  if [ -d /content ]; then EVALRX_ENV=/content/evalrx-env; else EVALRX_ENV="$HOME/evalrx-env"; fi
fi
PY_VERSION=3.12
STAMP="$EVALRX_ENV/.stamp"
log() { echo "[evalrx-bootstrap] $*"; }
die() { echo "[evalrx-bootstrap] ERROR: $*" >&2; exit 1; }

[ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || die "needs Linux x86_64, got $(uname -s)-$(uname -m)"

# Same lock + same source = nothing to do; a rerun of the cell is instant.
want_stamp="$ACCEL $(sha256sum "$LOCK" | cut -c1-16) $(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null || cat "$BUNDLE/COMMIT" 2>/dev/null || echo unknown)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want_stamp" ] && [ -x "$EVALRX_ENV/bin/evalrx" ]; then
  log "environment up to date: $EVALRX_ENV ($want_stamp)"
  exit 0
fi
# $EVALRX_ENV is itself the venv (bin/python, bin/evalrx, bin/agy); uv and the
# base Python live in hidden subdirectories of it.
rm -rf "$EVALRX_ENV"
mkdir -p "$EVALRX_ENV"
AGY_SRC=""

if [ -d "$BUNDLE" ]; then
  # ---------------------------------------------------------------- offline
  log "offline install from $BUNDLE"
  [ "$(cat "$BUNDLE/ACCEL")" = "$ACCEL" ] || die "bundle is for $(cat "$BUNDLE/ACCEL"), not $ACCEL"
  (cd "$BUNDLE" && sha256sum --quiet -c SHA256SUMS) || die "bundle checksum mismatch: re-download it"
  UV="$BUNDLE/uv"
  mkdir -p "$EVALRX_ENV/.python"
  tar -xzf "$BUNDLE/python.tar.gz" -C "$EVALRX_ENV/.python" --strip-components=1
  PYTHON="$EVALRX_ENV/.python/bin/python$PY_VERSION"
  INDEX_ARGS=(--offline --no-index --find-links "$BUNDLE/wheels")
  export UV_OFFLINE=1
  [ -f "$BUNDLE/agy" ] && AGY_SRC="$BUNDLE/agy"
else
  # ----------------------------------------------------------------- online
  pip_index="$(python3 -m pip config get global.index-url 2>/dev/null || true)"
  PYPI="${EVALRX_PYPI_INDEX:-${pip_index:-https://pypi.org/simple}}"
  if [ "$ACCEL" = gpu ]; then torch_default=https://download.pytorch.org/whl/cu129
  else torch_default=https://download.pytorch.org/whl/cpu; fi
  TORCH_INDEX="${EVALRX_TORCH_INDEX:-$torch_default}"
  PY_MIRROR="${EVALRX_PYTHON_MIRROR:-https://github.com/astral-sh/python-build-standalone/releases/download}"
  AGY_INSTALLER="${EVALRX_AGY_INSTALLER:-https://antigravity.google/cli/install.sh}"

  # Check every endpoint first, so a blocked network names what to override
  # instead of failing halfway through an install.
  probes=("EVALRX_PYPI_INDEX|${PYPI%/}/pip/" "EVALRX_TORCH_INDEX|${TORCH_INDEX%/}/torch/"
          "EVALRX_PYTHON_MIRROR|$PY_MIRROR")
  [ "${EVALRX_SKIP_AGY:-0}" = 1 ] || probes+=("EVALRX_AGY_INSTALLER|$AGY_INSTALLER")
  blocked=0
  for probe in "${probes[@]}"; do
    var="${probe%%|*}"; url="${probe#*|}"
    if python3 - "$url" <<'PY'
import sys, urllib.request, urllib.error
try:
    urllib.request.urlopen(urllib.request.Request(sys.argv[1], method="HEAD"), timeout=10)
except urllib.error.HTTPError as e:  # reachable; a mirror root may answer 403/404/405
    sys.exit(0 if e.code < 500 else 1)
except Exception as e:
    print(f"    {type(e).__name__}: {e}"[:200]); sys.exit(1)
PY
    then log "reachable  $url"
    else log "BLOCKED    $url  -> set $var to a reachable mirror"; blocked=1; fi
  done
  [ "$blocked" = 0 ] || die "network blocked; set the variables above, or use an offline bundle (tools/colab/build_bundle.sh, EVALRX_BUNDLE)"

  # uv through the kernel's pip, so a pip.conf mirror applies; into the env, not the kernel.
  python3 -m pip install -q --target "$EVALRX_ENV/.uv" --index-url "$PYPI" uv \
    || die "could not install uv from $PYPI (EVALRX_PYPI_INDEX)"
  UV="$EVALRX_ENV/.uv/bin/uv"
  export UV_PYTHON_INSTALL_MIRROR="$PY_MIRROR" UV_PYTHON_INSTALL_DIR="$EVALRX_ENV/.python"
  "$UV" python install -q --no-bin "$PY_VERSION" \
    || die "could not download Python $PY_VERSION from $PY_MIRROR (EVALRX_PYTHON_MIRROR)"
  PYTHON="$("$UV" python find --managed-python "$PY_VERSION")"
  INDEX_ARGS=(--index-url "$PYPI" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match)
  if [ "${EVALRX_SKIP_AGY:-0}" != 1 ]; then
    # A scratch HOME keeps the installer's PATH edits out of the user's ~/.bashrc.
    agy_tmp="$(mktemp -d)"
    python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
      "$AGY_INSTALLER" "$agy_tmp/install.sh"
    HOME="$agy_tmp" bash "$agy_tmp/install.sh" --dir "$agy_tmp/bin" >"$agy_tmp/install.log" 2>&1 \
      || { tail -n 20 "$agy_tmp/install.log"; die "agy install failed ($AGY_INSTALLER, EVALRX_AGY_INSTALLER); EVALRX_SKIP_AGY=1 skips it"; }
    AGY_SRC="$agy_tmp/bin/agy"
  fi
fi

log "Python: $PYTHON"
"$UV" venv -q --allow-existing --python "$PYTHON" "$EVALRX_ENV"
VPY="$EVALRX_ENV/bin/python"
"$UV" pip sync -q --python "$VPY" "${INDEX_ARGS[@]}" "$LOCK"
# Editable: the benchmark catalog is read from examples/ in this checkout.
"$UV" pip install -q --python "$VPY" "${INDEX_ARGS[@]}" --no-deps -e "$REPO_DIR"
[ -z "$AGY_SRC" ] || install -m 0755 "$AGY_SRC" "$EVALRX_ENV/bin/agy"

# agy reads its provider from settings; "gemini" selects GEMINI_API_KEY auth.
if [ -x "$EVALRX_ENV/bin/agy" ]; then
  "$VPY" - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".gemini/antigravity-cli/settings.json"
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg["modelProvider"] = "gemini"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(cfg, indent=2))
PY
fi

"$VPY" - "$ACCEL" <<'PY'
import importlib.metadata as md, sys
names = ["evalrx", "torch", "transformers"] if sys.argv[1] == "gpu" else ["evalrx", "jax", "libtpu", "gemma", "flax", "torch"]
print("[evalrx-bootstrap] versions", {n: md.version(n) for n in names}, "python", sys.version.split()[0])
PY
echo "$want_stamp" > "$STAMP"
log "ready: $EVALRX_ENV/bin (python, evalrx, agy)"
