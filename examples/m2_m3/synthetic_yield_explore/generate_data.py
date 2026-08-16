"""Generate a synthetic continuous-outcome dataset (chemical batch yield).

No model or API key needed. Demonstrates M2/M3 on a continuous outcome
(``yield_pct``), not just pass/fail logs — catalyst C and higher temperature
are seeded to produce a genuinely higher, tighter yield distribution so the
downstream explore run has real structure to find.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "data" / "batches.json"


def main() -> None:
    rng = random.Random(42)
    catalyst_offset = {"A": 0.0, "B": -1.5, "C": 2.5}
    rows = []
    for i in range(30):
        temperature = rng.uniform(150.0, 250.0)
        pressure = rng.uniform(3.0, 6.0)
        catalyst = rng.choice(["A", "B", "C"])
        base = 40.0 + 0.18 * temperature - 1.2 * pressure
        yield_pct = base + catalyst_offset[catalyst] + rng.gauss(0.0, 3.0)
        yield_pct = max(0.0, min(100.0, yield_pct))
        rows.append({
            "batch_id": f"b{i}",
            "temperature": round(temperature, 1),
            "pressure": round(pressure, 2),
            "catalyst": catalyst,
            "yield_pct": round(yield_pct, 1),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} synthetic batches to {OUT}")


if __name__ == "__main__":
    main()
