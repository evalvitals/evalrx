"""Append-only run records and resume.

Layout under the run's ``probing/`` directory::

    selection_trace.jsonl   one line per decision event (planned / settled / reused / skipped)
    cost_ledger.jsonl       one line per settled cost (the shared Stage-0 baseline first)
    observations/<id>.json  the full outcome of each terminal action attempt

Resume rule: an action whose identity already has a terminal observation is
read back, not re-run, and earns no new cost. An attempt that was planned but
never settled (the process died mid-action) is recorded as uncertain; the next
attempt gets a new number and its own cost line, so nothing is silently
counted twice or not at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from evalrx.eval_agent.probing.actions import TERMINAL, Observation, canonical


class Ledger:
    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)
        self.obs_dir = self.root / "observations"
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.root / "selection_trace.jsonl"
        self.cost_path = self.root / "cost_ledger.jsonl"

    # -- append-only logs ----------------------------------------------------
    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(canonical(record), ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a torn last line from a killed run
                    continue
        return out

    def trace(self, record: dict[str, Any]) -> None:
        self._append(self.trace_path, record)

    def cost(self, record: dict[str, Any]) -> None:
        self._append(self.cost_path, record)

    def trace_events(self) -> list[dict[str, Any]]:
        return self._read(self.trace_path)

    def cost_events(self) -> list[dict[str, Any]]:
        return self._read(self.cost_path)

    # -- observations ----------------------------------------------------------
    def save(self, obs: Observation) -> Path:
        path = self.obs_dir / f"{obs.action_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obs.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
        return path

    def settled(self, action_id: str) -> "Observation | None":
        path = self.obs_dir / f"{action_id}.json"
        if not path.is_file():
            return None
        obs = Observation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return obs if obs.status in TERMINAL else None

    def attempts(self, action_id: str) -> int:
        return sum(1 for e in self.trace_events()
                   if e.get("event") == "planned" and e.get("action_id") == action_id)

    def spent_wall_sec(self) -> float:
        return sum(float(e.get("wall_sec") or 0.0) for e in self.cost_events() if e.get("kind") == "action")
