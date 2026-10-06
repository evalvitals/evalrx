#!/usr/bin/env bash
# Install this checkout on a Colab CPU/TPU runtime. Run in a fresh runtime,
# before importing gemma/TensorFlow in a notebook cell. No CUDA wheels needed.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_DIR="${EVALRX_SETUP_DIR:-${HOME}/.cache/evalrx/colab}"
mkdir -p "$SETUP_DIR"
ver() { python3 -m pip show "$1" 2>/dev/null | sed -n 's/^Version: //p' || true; }
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Gemma requires Python 3.12 or newer"'

# Colab provides a matched JAX/libtpu set. Keep its CPU torch too: resolving
# torch afresh can download unnecessary CUDA packages into a TPU runtime.
# Keep fsspec compatible with Colab's preinstalled datasets package as well.
: > "$SETUP_DIR/constraints.txt"
for p in jax jaxlib libtpu torch fsspec; do
  v=$(ver "$p")
  if [ -n "$v" ]; then echo "$p==$v" >> "$SETUP_DIR/constraints.txt"; fi
done
if ! python3 -c 'import importlib.metadata as m; m.version("torch")' >/dev/null 2>&1; then
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  echo "torch==$(ver torch)" >> "$SETUP_DIR/constraints.txt"
fi
TF_BEFORE=""
for p in tensorflow tensorflow-cpu tensorflow-tpu; do
  v=$(ver "$p")
  if [ -n "$v" ]; then TF_BEFORE="$TF_BEFORE $p==$v"; fi
done
echo "Keeping runtime packages:"
cat "$SETUP_DIR/constraints.txt"

# Some Colab packages are installed by Debian without pip RECORD metadata.
# Shadow only a package that pip explicitly reports it cannot uninstall.
for attempt in 1 2 3 4; do
  if python3 -m pip install -c "$SETUP_DIR/constraints.txt" -e "$REPO_DIR[jax]" \
       'gemma==4.0.1' pytest gcsfs > "$SETUP_DIR/pip_setup.log" 2>&1; then
    break
  fi
  pkg=$(sed -n 's/.*Cannot uninstall \([A-Za-z0-9_.-]*\) .*/\1/p' "$SETUP_DIR/pip_setup.log" | head -1)
  if [ -z "$pkg" ] || [ "$attempt" = 4 ]; then tail -n 40 "$SETUP_DIR/pip_setup.log"; exit 1; fi
  echo "Shadowing distro-installed $pkg"
  python3 -m pip install --ignore-installed -c "$SETUP_DIR/constraints.txt" "$pkg" \
    > "$SETUP_DIR/pip_shadow.log" 2>&1 || { tail -n 20 "$SETUP_DIR/pip_shadow.log"; exit 1; }
done

# kauldron pulls tensorflow-cpu. If the runtime already had another TensorFlow
# distribution, restore that distribution's shared import directory afterwards.
for p in tensorflow tensorflow-cpu tensorflow-tpu; do
  v=$(ver "$p")
  if [ -n "$TF_BEFORE" ] && [ -n "$v" ] && [[ "$TF_BEFORE" != *"$p=="* ]]; then
    python3 -m pip uninstall -y "$p"
  fi
done
for spec in $TF_BEFORE; do
  python3 -m pip install --force-reinstall --no-deps "$spec" > "$SETUP_DIR/pip_tf.log" 2>&1 \
    || { tail -n 20 "$SETUP_DIR/pip_tf.log"; exit 1; }
done

# Fail visibly on broken imports or accelerator initialisation.
TF_CPP_MIN_LOG_LEVEL=2 python3 - <<'PY'
import tensorflow as tf
from gemma import gm
assert tf.io.gfile.exists('gs://gemma-data/tokenizers/tokenizer_gemma4.model')
print('SETUP tensorflow', tf.__version__, 'checkpoint bucket readable; gemma import OK')
PY
timeout -k 5 60 python3 -u - <<'PY'
import importlib.metadata as md
import jax
print('SETUP versions', {p: md.version(p) for p in ('evalrx', 'gemma', 'jax', 'jaxlib', 'flax', 'kauldron', 'torch')})
print('SETUP jax', jax.default_backend(), jax.devices())
PY
