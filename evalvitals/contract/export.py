"""Generate JSON Schema (and optionally TypeScript) from the Python contract.

    python -m evalvitals.contract.export --out docs/contract

Python stays the single source of truth; the schemas are build artifacts. Wire a
CI check that regenerates and fails on a diff, so a field added in Python can
never silently miss the frontend.

TypeScript:
    npx json-schema-to-typescript docs/contract/m1.ProbeOutput.schema.json \
        -o web/src/contract/ProbeOutput.d.ts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalvitals.contract import SCHEMA_VERSION, STAGE_IO


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stage, contract in STAGE_IO.items():
        for role, model in (("Input", contract.input), ("Output", contract.output)):
            schema = model.model_json_schema(mode="serialization")
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
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/contract"))
    args = ap.parse_args()
    for p in export(args.out):
        print(p)


if __name__ == "__main__":
    main()
