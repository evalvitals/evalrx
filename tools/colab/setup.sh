#!/usr/bin/env bash
# One-shot EvalRX jax_local setup on a Colab runtime (CPU or TPU), verified on
# the 2026-10-01 Colab image (Ubuntu 24.04, Python 3.13).
#   * keeps Colab's preinstalled jax / jaxlib / libtpu (a TPU runtime ships a
#     matching libtpu; letting pip move jax would break that pairing);
#   * kauldron (a gemma dependency) requires `tensorflow-cpu`; installed next to
#     Colab's `tensorflow` the two overwrite one directory and `import tensorflow`
#     fails, and etils.epath reads gs:// through tf.io.gfile, so every gs://
#     checkpoint / tokenizer read fails. Put the original TF back afterwards.
set -euo pipefail
REF="${EVALRX_REF:-ruinan}"
ver() { pip show "$1" 2>/dev/null | sed -n 's/^Version: //p' || true; }

: > /root/constraints.txt
for p in jax jaxlib libtpu; do v=$(ver $p); [ -n "$v" ] && echo "$p==$v" >> /root/constraints.txt; done
TF_BEFORE=""; for p in tensorflow tensorflow-cpu tensorflow-tpu; do v=$(ver $p); [ -n "$v" ] && TF_BEFORE="$TF_BEFORE $p==$v"; done
echo "pinned: $(tr '\n' ' ' < /root/constraints.txt)| tensorflow before:${TF_BEFORE:- none}"

# the TPU image ships Debian-installed packages (PyJWT ...) that pip cannot
# uninstall ("no RECORD file"); shadow each one into /usr/local and retry
for attempt in 1 2 3 4; do
  if pip install -q -c /root/constraints.txt "evalrx[jax] @ git+https://github.com/evalvitals/evalrx@${REF}" \
       "gemma==4.0.1" pytest > /root/pip_setup.log 2>&1; then
    break
  fi
  pkg=$(sed -n 's/.*Cannot uninstall \([A-Za-z0-9_.-]*\) .*/\1/p' /root/pip_setup.log | head -1)
  if [ -z "$pkg" ] || [ "$attempt" = 4 ]; then tail -n 30 /root/pip_setup.log; exit 1; fi
  echo "shadowing distro-installed $pkg"
  pip install -q --ignore-installed -c /root/constraints.txt "$pkg" > /root/pip_shadow.log 2>&1 || { tail -n 20 /root/pip_shadow.log; exit 1; }
done

for p in tensorflow tensorflow-cpu tensorflow-tpu; do
  v=$(ver $p)
  if [ -n "$TF_BEFORE" ] && [ -n "$v" ] && [[ "$TF_BEFORE" != *"$p=="* ]]; then
    echo "removing $p==$v pulled in next to the runtime's TensorFlow"
    pip uninstall -y -q "$p"
  fi
done
for spec in $TF_BEFORE; do
  echo "restoring $spec"
  pip install -q --force-reinstall --no-deps "$spec" > /root/pip_tf.log 2>&1 || { tail -n 20 /root/pip_tf.log; exit 1; }
done

[ -d /root/evalrx ] || git clone -q --depth 1 -b "$REF" https://github.com/evalvitals/evalrx.git /root/evalrx

# separate processes: TF must import cleanly (gs:// reads), jax must see the accelerator
python3 -c "import tensorflow as tf; print('SETUP tf', tf.__version__, 'gs:// readable:', tf.io.gfile.exists('gs://gemma-data/tokenizers/tokenizer_gemma4.model'))" 2>&1 \
  | grep -E "^(SETUP|\w+Error)" || true
timeout -k 5 120 python3 -c "
import importlib.metadata as md, jax
print('SETUP versions', {p: md.version(p) for p in ('evalrx', 'gemma', 'jax', 'jaxlib', 'flax', 'kauldron')})
print('SETUP jax', jax.default_backend(), jax.devices())" 2>&1 | grep -E "^(SETUP|\w+Error)" || true
