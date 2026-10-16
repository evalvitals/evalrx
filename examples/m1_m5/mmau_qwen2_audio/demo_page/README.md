# Standalone MMAU demo artifact

`index.html` is a generated, self-contained report with embedded media:
95,617,127 bytes (91.2 MiB). It stays at this path on existing workspaces and
is ignored by Git. New source snapshots contain these recovery instructions
instead of the generated HTML.

From the repository root, restore the exact historical report:

```bash
python tools/restore_mmau_demo.py
```

The script first reads the pinned blob from local Git history. A shallow clone
or source archive can download it from the same immutable revision with
`--source http`. Both paths verify the byte count and SHA-256 digest before
publishing the file. An existing file is verified and never overwritten.

- Source revision: `770947992e32e3dbd2b12ed4c26047941bc5b175`
- Source path: `examples/m1_m5/mmau_qwen2_audio/demo_page/index.html`
- SHA-256: `005cfe088dcecdd3159fd318d9cd6473ccdc299464771cc3634011393ab1958b`

Alternatively, export a new report from your own local run using the
[example's export command](../README.md#demo-page-the-whole-loop-as-one-shareable-html-file).
That produces a new artifact; it need not match this historical checksum.

Git history has not been rewritten. To avoid downloading historical generated
files initially, use a shallow clone and restore this artifact only if needed.
