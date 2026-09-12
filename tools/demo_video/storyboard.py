#!/usr/bin/env python3
"""Build a demo-video storyboard from a real EvalRX run.

Two input shapes, both real runs — nothing here invents numbers:

* ``--run-dir DIR`` — a live RunLoggerV2 output (``logs/run.json`` +
  ``logs/M*/log.json``). Every narrated line comes from the real
  ``LoopNarrator`` fed the logged records, exactly as
  ``examples/benchmark/tools/replay_narrate.py`` does, so durations are the
  ones the stages actually took.

* ``--report FILE.html`` — an exported report page (``docs/demo/*.html``),
  which embeds ``window.__EVALRX_REPORT__``. Use this when the run directory
  is not at hand. The report keeps the event stream in compacted form: each
  event has its real timestamp but not the per-record fields the narrator
  reads, so stage **durations are recomputed as timestamp deltas** and the
  counts (analyzers, findings, hypotheses, verdicts) are taken from the
  report's own ``stages``/``stage_detail``. Same facts, reassembled — the
  storyboard records which path produced it in ``meta.provenance``.

The output is a JSON storyboard consumed by ``render.py``; keeping the two
apart means the visual pass can be re-run without touching a model, and a new
run only has to be re-extracted, not re-designed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

# The narrator's own geometry (evalrx/eval_agent/narration.py), duplicated here
# so a storyboard can be built without importing evalrx.
LABEL_WIDTH = 12
STAGE_LABELS = {"M1": "probe", "M2": "explore", "M3": "diagnose",
                "M4": "verify", "M5": "intervene"}
RUN_LABEL = "loop"
NARRATED = ("probe", "analysis", "explore", "diagnosis", "surgery", "experiment", "fix")


def row(code: str, detail: str) -> dict[str, Any]:
    """One narrated line, split into the parts the renderers colour apart."""
    label = STAGE_LABELS.get(code, RUN_LABEL if code == "RUN" else code.lower())
    dots = "·" * max(4, LABEL_WIDTH + 2 - len(label))
    return {"kind": "row", "code": code.ljust(3), "label": label.ljust(LABEL_WIDTH),
            "dots": dots, "detail": detail}


def _ts(event: dict[str, Any]) -> dt.datetime:
    return dt.datetime.fromisoformat(event["ts"])


def _timing(seconds: float | None) -> str:
    return f" ({seconds:.1f}s)" if seconds is not None else ""


def _cycle(event: dict[str, Any]) -> str:
    cycle = event.get("cycle")
    return f"cycle {cycle} · " if cycle not in (None, -1) else ""


def from_report(path: Path) -> dict[str, Any]:
    """Storyboard from an exported report page's embedded report data."""
    src = path.read_text(errors="replace")
    marker = "window.__EVALRX_REPORT__="
    at = src.rindex(marker) + len(marker)
    end = src.index("</script>", at)
    payload = json.loads(src[at:end].rstrip().rstrip(";"))
    data = payload["data"]

    setting = data.get("setting") or {}
    metrics = {m["id"]: m["value"] for m in data.get("metrics") or []}
    stages = {s["code"]: s for s in data.get("stages") or []}
    detail = data.get("stage_detail") or {}
    events = [e for e in (data.get("debug") or {}).get("events") or []
              if e.get("event") in NARRATED + ("run_start", "loop_end")]

    n_cases = int(setting.get("n_cases") or metrics.get("evaluated") or 0)
    n_failed = int(metrics.get("failed") or 0)
    model = setting.get("model") or "model"
    dataset_full = setting.get("dataset") or "dataset"
    dataset = re.sub(r"\s*\(.*\)$", "", dataset_full)

    beats: list[dict[str, Any]] = []
    beats.append({"kind": "plain", "text": f"[model] {model}; dataset={dataset.lower()}; "
                                           f"n={n_cases}", "cls": "dim"})
    if n_failed:
        acc = (n_cases - n_failed) / n_cases if n_cases else 0.0
        beats.append({"kind": "plain", "cls": "dim",
                      "text": f"Baseline: PASS={n_cases - n_failed}, FAIL={n_failed}, "
                              f"accuracy={acc:.4f}"})
    beats.append({"kind": "plain", "text": f"VLDiagnoseLoop  model={model}  "
                                           f"held-out confirmation ON", "cls": "banner"})

    prev = _ts(events[0]) if events else None
    n_hyp = int((stages.get("M3") or {}).get("evidence_count") or 0)
    verdicts = [r.get("status") for r in (detail.get("m4") or {}).get("results") or []]
    verdict_text = [
        (r.get("plain_statement") or r.get("statement") or "").strip()
        for r in (detail.get("m3") or {}).get("hypotheses") or []
    ]
    seen_surgery = 0

    t_zero = _ts(events[0]) if events else None
    for event in events:
        key = event.get("event")
        code = event.get("stage") or "RUN"
        gap = (_ts(event) - prev).total_seconds() if prev else None
        prev = _ts(event)
        mark_from = len(beats)

        if key == "run_start":
            beats.append(row("RUN", f"starting · model={model} · n_cases={n_cases}"))
        elif key == "probe":
            n = int((stages.get("M1") or {}).get("evidence_count") or 0)
            beats.append(row(code, f"{_cycle(event)}{n} analyzer{'s' if n != 1 else ''} "
                                   f"run{_timing(gap)}"))
        elif key == "explore":
            beats.append(row(code, f"{_cycle(event)}explore report ready{_timing(gap)}"))
        elif key == "analysis":
            n = int((stages.get("M2") or {}).get("evidence_count") or 0)
            beats.append(row(code, f"{_cycle(event)}{n} finding(s){_timing(gap)}"))
        elif key == "diagnosis":
            noun = "hypothesis" if n_hyp == 1 else "hypotheses"
            beats.append(row(code, f"{_cycle(event)}{n_hyp} falsifiable {noun} "
                                   f"proposed{_timing(gap)}"))
        elif key == "surgery":
            status = (event.get("summary") or "").strip() or "inconclusive"
            mark = "ok" if status == "supported" else "bad"
            claim = verdict_text[seen_surgery] if seen_surgery < len(verdict_text) else ""
            seen_surgery += 1
            beats.append({**row(code, f"{_cycle(event)}{{mark}} {claim[:58]}… — {status}"
                                      f"{_timing(gap)}"), "mark": mark})
        elif key == "loop_end":
            stopped = (event.get("summary") or "").strip()
            n_ok = sum(1 for v in verdicts if v == "supported")
            beats.append(row("RUN", f"done · resolved={'True' if n_ok else 'False'} · "
                                    f"stopped_by={stopped} · {n_ok} supported"))
        elif key == "experiment":
            status = (event.get("summary") or "").strip()
            beats.append({**row(code, f"{{mark}} repair experiment — {status}{_timing(gap)}"),
                          "mark": "ok" if status == "supported" else "bad"})
        elif key == "fix":
            m5 = detail.get("m5") or {}
            best = m5.get("best") or {}
            if m5.get("fixed"):
                beats.append({**row(code, f"{{mark}} {best.get('name')} Δ{best.get('effect'):+.3f} "
                                          f"(n_fixed={best.get('n_fixed')}, "
                                          f"n_broken={best.get('n_broken')})"), "mark": "ok"})
            else:
                rec = m5.get("recommendation") or {}
                cand = (m5.get("candidates") or [{}])[0]
                beats.append({**row(code, f"{{mark}} {cand.get('name') or 'candidate'} did not "
                                          f"validate — recommend {rec.get('recommend_tier') or '—'}"),
                              "mark": "bad"})

        # Real timings ride along with every beat this event produced, so the
        # renderers can pace the reveal by what the stage actually cost and
        # show a real elapsed clock.
        for beat in beats[mark_from:]:
            beat["real_sec"] = gap
            beat["elapsed_sec"] = ((_ts(event) - t_zero).total_seconds()
                                   if t_zero is not None else None)

    return {
        "meta": {
            "model": model,
            "dataset": dataset,
            "dataset_full": dataset_full,
            "n_cases": n_cases,
            "n_failed": n_failed,
            "trace_id": data.get("trace_id"),
            "source": str(path),
            "provenance": "exported report page (window.__EVALRX_REPORT__); stage durations "
                          "are event-timestamp deltas, counts from the report's own stage data",
            "fixed": bool((detail.get("m5") or {}).get("fixed")),
            "n_supported": sum(1 for v in verdicts if v == "supported"),
            "n_hypotheses": n_hyp,
            "wall_clock_sec": ((_ts(events[-1]) - _ts(events[0])).total_seconds()
                               if len(events) > 1 else 0.0),
        },
        "beats": beats,
    }


def from_run_dir(path: Path) -> dict[str, Any]:
    """Storyboard from a live run directory, via the real LoopNarrator."""
    import io
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evalrx.eval_agent.narration import LoopNarrator  # noqa: E402

    logs = path / "logs" if (path / "logs" / "run.json").exists() else path
    run = json.loads((logs / "run.json").read_text())
    buffer = io.StringIO()
    narrator = LoopNarrator(stream=buffer, color=False)

    start = run.get("run_start") or {}
    narrator.on_run_start(start) if hasattr(narrator, "on_run_start") else None
    records: list[tuple[int, str, str, dict[str, Any]]] = []
    for stage in ("M1", "M2", "M3", "M4", "M5"):
        log = logs / stage / "log.json"
        if not log.exists():
            continue
        document = json.loads(log.read_text())
        for key in NARRATED:
            for record in document.get(key) or []:
                records.append((int(record.get("event_seq") or 0), stage, key, record))
    for _, stage, key, record in sorted(records):
        narrator.on_event(stage, key, record)

    beats = [{"kind": "plain", "text": line, "cls": "dim"}
             for line in buffer.getvalue().splitlines() if line.strip()]
    return {
        "meta": {
            "model": start.get("model") or "model",
            "dataset": start.get("benchmark_name") or "dataset",
            "n_cases": start.get("n_cases") or 0,
            "source": str(path),
            "provenance": "live run directory, rendered by the real LoopNarrator",
        },
        "beats": beats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path, help="exported report .html")
    source.add_argument("--run-dir", type=Path, help="RunLoggerV2 run directory")
    ap.add_argument("--out", type=Path, required=True, help="storyboard .json to write")
    args = ap.parse_args()

    board = from_report(args.report) if args.report else from_run_dir(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.out}  ({len(board['beats'])} beats, {board['meta']['provenance']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
