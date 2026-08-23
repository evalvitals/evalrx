"""Generate JSON Schema (and optionally TypeScript) from the Python contract.

    python -m evalvitals.contract.export --out docs/contract

Python stays the single source of truth; the schemas are build artifacts. Wire a
CI check that regenerates and fails on a diff, so a field added in Python can
never silently miss the frontend.

TypeScript is generated here too (``contract.d.ts``), in Python — see
:mod:`evalvitals.contract.typescript` for why not npx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalvitals.contract import SCHEMA_VERSION, STAGE_IO


def export(out_dir: Path, *, typescript: bool = True, frontend: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    schemas: dict[str, dict] = {}
    for stage, contract in STAGE_IO.items():
        for role, model in (("Input", contract.input), ("Output", contract.output)):
            schema = model.model_json_schema(mode="serialization")
            schemas[f"{stage}.{model.__name__}"] = schema
            schema["$id"] = f"https://evalvitals.dev/contract/{stage}.{model.__name__}.json"
            schema["x-stage"] = stage
            schema["x-role"] = role.lower()
            schema["x-schema-version"] = SCHEMA_VERSION
            schema["x-optional-stage"] = contract.optional
            path = out_dir / f"{stage}.{model.__name__}.schema.json"
            path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
            written.append(path)

    index = {
        "schema_version": SCHEMA_VERSION,
        "stages": [
            {
                "stage": s,
                "purpose": c.purpose,
                "optional": c.optional,
                "input": f"{s}.{c.input.__name__}.schema.json",
                "output": f"{s}.{c.output.__name__}.schema.json",
            }
            for s, c in STAGE_IO.items()
        ],
    }
    idx = out_dir / "index.json"
    idx.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    written.append(idx)

    if typescript:
        from typing import get_args

        from evalvitals.contract.common import MEDIA_SLOTS, Modality
        from evalvitals.contract.typescript import render

        rendered = render(
            schemas, schema_version=SCHEMA_VERSION,
            aliases={"Modality": get_args(Modality), "MediaSlot": MEDIA_SLOTS},
        )
        targets = [out_dir / "contract.d.ts"]
        if frontend:
            targets += _frontend_targets()
        for ts in targets:
            ts.parent.mkdir(parents=True, exist_ok=True)
            ts.write_text(rendered)
            written.append(ts)
    return written


def _frontend_targets() -> "list[Path]":
    """Where else the declaration file has to land.

    The report UI imports these types, so it needs the file inside its own
    ``src/``. Writing it from here — rather than copying it in by hand — is what
    keeps the two from drifting: one command regenerates both, and ``--check``
    fails on either being stale. A vendored copy nobody regenerates is worse
    than no types at all, because it looks authoritative while being wrong.
    """
    web = Path(__file__).resolve().parents[1] / "reporting" / "web" / "src" / "contract"
    return [web / "contract.d.ts"] if web.parent.parent.is_dir() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/contract"))
    ap.add_argument("--no-typescript", action="store_true", help="JSON Schema only.")
    ap.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if any output differs from what is on disk. For CI: a "
             "generated artifact nothing verifies is a stale artifact.",
    )
    args = ap.parse_args()

    if args.check:
        import tempfile

        # frontend=False: a check must not write into the tree it is checking,
        # or the second run always passes.
        with tempfile.TemporaryDirectory() as tmp:
            fresh = export(Path(tmp), typescript=not args.no_typescript, frontend=False)
            stale = []
            for f in fresh:
                targets = [args.out / f.name]
                if f.name == "contract.d.ts":
                    targets += _frontend_targets()
                for target in targets:
                    if not target.exists() or target.read_text() != f.read_text():
                        stale.append(str(target))
        if stale:
            raise SystemExit(
                "contract artifacts are out of date: " + ", ".join(sorted(stale))
                + f"\nregenerate with: python -m evalvitals.contract.export --out {args.out}"
            )
        print(f"{args.out}: up to date")
        return

    for p in export(args.out, typescript=not args.no_typescript):
        print(p)


if __name__ == "__main__":
    main()
