"""DiagnosticTracer — Native Langfuse & OpenTelemetry Tracing Engine for EvalRX.

This module provides first-class observability for model evaluations and agentic diagnostics:
1. Live streaming to a Langfuse server (self-hosted or cloud) via the Langfuse Python SDK (v3/v4 API).
2. Standardized JSON Trace Bundle export (OpenTelemetry-compatible) for offline auditing —
   this file is the durable source of truth and works with no SDK installed.
3. Multi-tier hierarchical span recording:
   - Root Trace: Benchmark run session (model metadata, benchmark parameters, dataset fingerprint).
   - Pipeline Stage Spans: PRE-M1, M1 Checkup, M2 Screening, M3 Diagnosis, M4 Adjudication, M5 Repair,
     plus Explore and Tool-Codegen (the coder-agent trajectories).
   - Probe Execution Spans: Granular probe runs with findings and artifact paths.
   - AI Doctor Generations: LLM reasoning, screening evidence digest, and falsifiable hypothesis generation.
   - Repair Generations: Candidate patch searches, prompt templates, and McNemar confirmation scores.

Live-sync contract:
- Set ``EVALRX_LANGFUSE_MODE=live`` plus LANGFUSE_PUBLIC_KEY /
  LANGFUSE_SECRET_KEY (and LANGFUSE_HOST for self-hosted) for a user-facing
  run. Missing credentials then fail fast. ``auto`` retains compatibility for
  local development, and ``offline`` is an explicit no-upload choice.
- The Langfuse trace id is the RunLogger trace_id (a UUID, dashes stripped) so the live trace,
  the exported bundle and run_log.jsonl all agree on identity.
- Every live failure is reported ONCE on stderr instead of being silently swallowed — an
  observability layer that fails quietly is worse than none.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from evalrx.observability.envelope import artifact_manifests_for_event, make_event_envelope
from evalrx.observability.outbox import ObservabilityOutbox


class DiagnosticTracer:
    """Manages active tracing context across the diagnostic pipeline.

    All records are accumulated in memory and exported verbatim via
    :meth:`export_bundle`; the optional Langfuse client mirrors them live.
    """

    def __init__(
        self,
        run_dir: str | Path | None = None,
        auto_sync: bool = False,
        mode: str | None = None,
    ):
        self.run_dir = Path(run_dir).resolve() if run_dir else None
        self.auto_sync = auto_sync
        self.trace_id = f"evalrx_{uuid.uuid4().hex[:12]}"
        self.spans: list[dict[str, Any]] = []
        self.generations: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.trace_metadata: dict[str, Any] = {}
        self._start_time = time.time()
        requested_mode = (mode or os.getenv("EVALRX_LANGFUSE_MODE", "auto")).strip().lower()
        if requested_mode not in {"auto", "live", "offline"}:
            raise ValueError("EVALRX_LANGFUSE_MODE must be one of: auto, live, offline")
        self.mode = requested_mode
        self._langfuse_client = None
        self._live_root = None          # root "chain" observation for the run
        self._live_obs: dict[str, Any] = {}  # our span_id -> live observation wrapper
        self._warned: set[str] = set()
        # Every structured run-log event enters this durable queue before live
        # delivery.  It is intentionally kept separate from the human-readable
        # JSONL log so delivery retries never mutate the run record.
        self.outbox = ObservabilityOutbox(
            (self.run_dir / ".evalrx" / "langfuse_outbox.sqlite3")
            if self.run_dir is not None
            else Path(".evalrx-langfuse-outbox.sqlite3")
        )

        has_credentials = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        if requested_mode == "live" and not has_credentials:
            raise RuntimeError(
                "Live Langfuse was requested but LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY are not configured. Set EVALRX_LANGFUSE_MODE=offline "
                "only for an explicitly offline run."
            )
        if requested_mode != "offline" and has_credentials:
            try:
                from langfuse import Langfuse

                self._langfuse_client = Langfuse()
            except Exception as exc:
                self._warn(
                    "langfuse_init",
                    f"LANGFUSE keys are set but the Langfuse client failed to initialize; "
                    f"continuing with local-only tracing. Error: {exc}",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warn(self, key: str, msg: str) -> None:
        """Report a live-sync problem once per key; silent afterwards."""
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[langfuse] {msg}", file=sys.stderr)

    @property
    def langfuse_trace_id(self) -> str:
        """Langfuse requires a 32-hex trace id; our RunLogger trace_id is a UUID."""
        return self.trace_id.replace("-", "")

    def _apply_tags(self, tags: list[str]) -> None:
        """Best-effort trace tags (private SDK helper; cosmetic if unavailable)."""
        if self._langfuse_client is None:
            return
        try:
            fn = getattr(self._langfuse_client, "_create_trace_tags_via_ingestion", None)
            if fn is not None:
                fn(trace_id=self.langfuse_trace_id, tags=[str(t) for t in tags if t])
        except Exception:
            pass  # tags are cosmetic; never fail the run on them

    def record_event(self, event: dict[str, Any], *, event_seq: int) -> None:
        """Queue one complete EvalRX event for durable Langfuse delivery.

        This is called after the JSONL write has received its timestamp and
        trace id.  The deterministic event id makes repeated process startup
        and manual backfill safe.
        """
        artifact_refs = (
            artifact_manifests_for_event(event, run_dir=self.run_dir)
            if self.run_dir is not None
            else []
        )
        envelope = make_event_envelope(
            event, trace_id=self.trace_id, event_seq=event_seq, artifact_refs=artifact_refs,
        )
        self.events.append(envelope)
        self.outbox.enqueue(envelope)
        # A live diagnostic should be inspectable while it is running.  The
        # SQLite outbox remains the durability boundary: a transient upload
        # failure simply leaves this event pending for the next stage/close.
        if self.auto_sync and self._langfuse_client is not None:
            self.flush()

    def _publish_event(self, envelope: dict[str, Any]) -> None:
        """Publish a queued event as a native Langfuse EVENT observation."""
        if self._langfuse_client is None:
            raise RuntimeError("Langfuse client is not configured")
        # Attach audit events to the run chain rather than emitting a flat trace
        # row.  The deterministic event id still makes retry/backfill safe.
        input_data: dict[str, Any] = {"event": envelope["payload"]}
        media = self._media_payload(envelope)
        if media:
            input_data["artifacts"] = media
        metadata = {
            "evalrx_schema_version": envelope["schema_version"],
            "event_id": envelope["event_id"],
            "event_seq": envelope["event_seq"],
            "stage": envelope["stage"],
            "cycle": envelope.get("cycle"),
            "span_id": envelope["payload"].get("span_id"),
            "artifact_refs": envelope["artifact_refs"],
        }
        parent = self._live_root
        if parent is not None:
            observation = parent.start_observation(
                name=f"EvalRX {envelope['stage']}: {envelope['event_type']}",
                as_type="event", input=input_data, metadata=metadata,
            )
            observation.end()
            return
        self._langfuse_client.create_event(
            trace_context={"trace_id": self.langfuse_trace_id},
            name=f"EvalRX {envelope['stage']}: {envelope['event_type']}",
            input=input_data, metadata=metadata,
        )

    def _media_payload(self, envelope: dict[str, Any]) -> list[dict[str, Any]]:
        """Attach supported files as Langfuse Media while retaining every manifest.

        Non-displayable files (for example ``.npy`` tensors) are uploaded as
        ``application/octet-stream``.  They still remain downloadable and
        checksum-addressable even when the Langfuse UI cannot preview them.
        """
        if self.run_dir is None:
            return []
        try:
            from langfuse import LangfuseMedia
            from langfuse.api.media.types.media_content_type import MediaContentType
        except ImportError:
            return []
        result: list[dict[str, Any]] = []
        for manifest in envelope["artifact_refs"]:
            path = self.run_dir / str(manifest["path"])
            if path.is_file():
                try:
                    content_type = MediaContentType(str(manifest["mime_type"]))
                except ValueError:
                    content_type = MediaContentType.APPLICATION_OCTET_STREAM
                result.append({
                    "role": manifest["role"],
                    "artifact_id": manifest["artifact_id"],
                    "content": LangfuseMedia(file_path=str(path), content_type=content_type),
                })
        return result

    # ------------------------------------------------------------------
    # Trace / span / generation / score recording
    # ------------------------------------------------------------------

    def start_trace(self, model: str, benchmark: str, n_cases: int = 0, metadata: dict[str, Any] | None = None) -> None:
        """Initialize the root trace."""
        self.trace_metadata = {
            "model": model,
            "benchmark": benchmark,
            "n_cases": n_cases,
            "created_at": time.time(),
            **(metadata or {}),
        }
        if self._langfuse_client is not None:
            try:
                self._live_root = self._langfuse_client.start_observation(
                    trace_context={"trace_id": self.langfuse_trace_id},
                    name=f"EvalRX: {model} on {benchmark}",
                    as_type="chain",
                    input=self.trace_metadata,
                    metadata=self.trace_metadata,
                )
                self._apply_tags(["evalrx", model, benchmark])
            except Exception as exc:
                self._warn("start_trace", f"failed to create Langfuse root trace: {exc}")

    def start_span(
        self,
        name: str,
        stage: str,
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> str:
        """Create a new pipeline or probe span (nested under *parent_id* when given)."""
        span_id = f"{self.trace_id}_{stage.lower()}_{uuid.uuid4().hex[:6]}"
        span_rec = {
            "id": span_id,
            "trace_id": self.trace_id,
            "parent_id": parent_id,
            "name": name,
            "stage": stage,
            "start_time": time.time(),
            "input": input_data,
            "metadata": metadata or {},
            "status": "running",
        }
        self.spans.append(span_rec)
        if self._langfuse_client is not None:
            try:
                parent_obs = self._live_obs.get(parent_id) if parent_id else None
                if parent_obs is None:
                    parent_obs = self._live_root
                if parent_obs is not None:
                    obs = parent_obs.start_observation(
                        name=name, input=input_data, metadata=metadata or {}
                    )
                else:
                    obs = self._langfuse_client.start_observation(
                        trace_context={"trace_id": self.langfuse_trace_id},
                        name=name,
                        input=input_data,
                        metadata=metadata or {},
                    )
                self._live_obs[span_id] = obs
            except Exception as exc:
                self._warn("start_span", f"failed to create Langfuse span {name!r}: {exc}")
        return span_id

    def end_span(self, span_id: str, output_data: Any = None, status: str = "completed") -> None:
        """Mark a span as ended (locally and, when live, on Langfuse)."""
        for s in self.spans:
            if s["id"] == span_id:
                s["end_time"] = time.time()
                s["duration_sec"] = s["end_time"] - s["start_time"]
                s["output"] = output_data
                s["status"] = status
                break
        obs = self._live_obs.pop(span_id, None)
        if obs is not None:
            try:
                if output_data is not None:
                    obs.update(output=output_data)
                obs.end()
            except Exception as exc:
                self._warn("end_span", f"failed to end Langfuse span {span_id!r}: {exc}")

    def log_generation(
        self,
        name: str,
        model: str,
        prompt: Any,
        completion: Any,
        span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> str:
        """Record an LLM agent / diagnostician generation."""
        gen_id = f"gen_{uuid.uuid4().hex[:8]}"
        gen_rec = {
            "id": gen_id,
            "trace_id": self.trace_id,
            "span_id": span_id,
            "name": name,
            "model": model,
            "prompt": prompt,
            "completion": completion,
            "usage": usage or {},
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.generations.append(gen_rec)
        if self._langfuse_client is not None:
            try:
                parent = self._live_obs.get(span_id) if span_id else None
                if parent is None:
                    parent = self._live_root
                if parent is not None:
                    gen = parent.start_observation(
                        name=name,
                        as_type="generation",
                        model=model,
                        input=prompt,
                        output=completion,
                        metadata=metadata or {},
                    )
                else:
                    gen = self._langfuse_client.start_observation(
                        trace_context={"trace_id": self.langfuse_trace_id},
                        name=name,
                        as_type="generation",
                        model=model,
                        input=prompt,
                        output=completion,
                        metadata=metadata or {},
                    )
                gen.end()
            except Exception as exc:
                self._warn("log_generation", f"failed to create Langfuse generation {name!r}: {exc}")
        return gen_id

    def log_score(
        self,
        name: str,
        value: float,
        comment: str = "",
        span_id: str | None = None,
    ) -> None:
        """Attach a quantitative metric or evaluation score to trace/span."""
        score_rec = {
            "name": name,
            "value": float(value),
            "comment": comment,
            "span_id": span_id,
            "timestamp": time.time(),
        }
        self.scores.append(score_rec)
        if self._langfuse_client is not None:
            obs = self._live_obs.get(span_id) if span_id else None
            try:
                if obs is not None:
                    obs.score(name=name, value=float(value), comment=comment)
                else:
                    self._langfuse_client.create_score(
                        trace_id=self.langfuse_trace_id,
                        name=name,
                        value=float(value),
                        comment=comment,
                    )
            except Exception as exc:
                self._warn("log_score", f"failed to create Langfuse score {name!r}: {exc}")

    def export_bundle(self, out_path: str | Path | None = None) -> dict[str, Any]:
        """Export the full trace hierarchy to a clean JSON bundle."""
        bundle = {
            "trace": {
                "id": self.trace_id,
                "name": f"EvalRX: {self.trace_metadata.get('model', 'Model')}",
                "metadata": self.trace_metadata,
                "duration_sec": time.time() - self._start_time,
            },
            "spans": self.spans,
            "generations": self.generations,
            "scores": self.scores,
            "events": self.events,
        }
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return bundle

    def flush(self) -> None:
        """Flush any queued live observations to Langfuse."""
        if self._langfuse_client is not None:
            try:
                published, failed = self.outbox.drain(self._publish_event)
                if failed:
                    self._warn(
                        "outbox_delivery",
                        f"{failed} Langfuse event(s) remain in the local outbox for retry.",
                    )
                self._langfuse_client.flush()
            except Exception as exc:
                self._warn("flush", f"failed to flush Langfuse client: {exc}")

    @property
    def live_enabled(self) -> bool:
        """Whether this run has an active Langfuse client (safe for UI provenance)."""
        return self._langfuse_client is not None

    def end_trace(self, output_data: Any = None) -> None:
        """Finish the live root observation, if live mirroring was enabled."""
        root, self._live_root = self._live_root, None
        if root is None:
            return
        try:
            if output_data is not None:
                root.update(output=output_data)
            root.end()
        except Exception as exc:
            self._warn("end_trace", f"failed to end Langfuse root trace: {exc}")


# ---------------------------------------------------------------------------
# Offline (run-directory) bundle export & batch sync
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> "float | None":
    """Parse a headline value like "12.3%", 12.3 or "1,234" into a float."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().rstrip("%").replace(",", "")
        return float(s)
    except (TypeError, ValueError):
        return None


def _resolve_trace_id(run_dir: Path, fingerprint: str) -> str:
    """The run's trace id — the SAME uuid used by run_log.jsonl and langfuse_trace.json.

    Resolution order: langfuse_trace.json (written live by RunLogger) → the first
    run_start event in run_log.jsonl → a deterministic 32-hex id derived from the
    data fingerprint (Langfuse requires uuid-shaped trace ids).
    """
    import hashlib

    bundle_path = run_dir / "langfuse_trace.json"
    if bundle_path.exists():
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            tid = (bundle.get("trace") or {}).get("id")
            if tid:
                return str(tid)
        except Exception:
            pass
    for candidate in (run_dir / "run_log.jsonl", run_dir / "logs" / "run_log.jsonl"):
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    if ev.get("event") == "run_start" and ev.get("trace_id"):
                        return str(ev["trace_id"])
                    break  # only the first (pre-reload) run_start counts
            except Exception:
                pass
            break
    return hashlib.md5((fingerprint or "evalrx").encode("utf-8")).hexdigest()


def export_to_langfuse_bundle(run_dir: str | Path, out_json: str | Path | None = None) -> dict[str, Any]:
    """Convert an EvalRX run artifacts directory into a rich Langfuse bundle."""
    from evalrx.reporting.html_report import extract_run_data

    run_dir = Path(run_dir).resolve()
    data = extract_run_data(run_dir)
    run = data["run"]
    m1 = data["m1"]
    m2 = data["m2"]
    m3 = data["m3"]
    m4 = data["m4"]
    m5_s = data["m5_surgery"]
    m5_f = data["m5_fix"]

    trace_id = _resolve_trace_id(run_dir, run.get("data_fingerprint") or "")
    trace_name = f"EvalRX: {run['model']} · {run['benchmark_name']}"

    trace = {
        "id": trace_id,
        "name": trace_name,
        "release": f"v{run['version']}",
        "tags": ["evalrx", run["model"], run["benchmark_name"], run["stopped_by"]],
        "metadata": {
            "model": run["model"],
            "raw_model": run["raw_model"],
            "benchmark": run["benchmark_name"],
            "n_cases": run["n_cases"],
            "cycles": run["cycles"],
            "protocol": run["protocol"],
            "data_fingerprint": run["data_fingerprint"],
            "logs_dir": run["logs_dir"],
        },
    }

    spans = []
    generations = []
    scores = []

    # PRE-M1 Span
    if data["pre_m1"]["ran"]:
        spans.append({
            "id": f"{trace_id}_pre_m1",
            "name": "PRE-M1: Case Synthesis & Adversarial Probing",
            "type": "span",
            "metadata": {"stage": "PRE-M1", "n_synthesized": data["pre_m1"]["n_cases"]},
            "input": {"criteria": "adversarial_blindspots"},
            "output": {"synthesized_cases_count": data["pre_m1"]["n_cases"]},
        })

    # M1 Span & Sub-spans for Probes
    m1_span_id = f"{trace_id}_m1"
    spans.append({
        "id": m1_span_id,
        "name": "M1: Multi-Dimensional Checkup & Signals",
        "type": "span",
        "metadata": {
            "stage": "M1",
            "analyzers": m1["analyzers"],
            "duration_sec": m1["duration"],
            "n_probes": len(m1["results"]),
        },
        "input": {"analyzers": m1["analyzers"]},
        "output": {"summary": [f"{r['name']}: {r['display_name']}" for r in m1["results"]]},
    })

    # Individual Probe Sub-Spans
    for r in m1["results"]:
        probe_span_id = f"{m1_span_id}_{r['name']}"
        findings = r.get("findings") or {}
        spans.append({
            "id": probe_span_id,
            "parent_id": m1_span_id,
            "name": f"Probe: {r['display_name']} ({r['name']})",
            "type": "span",
            "metadata": {
                "probe_code": r["name"],
                "question": r["question"],
                "n_scored": r["n"],
                "headlines": r["headline"],
            },
            "input": {"probe_name": r["name"], "question": r["question"]},
            "output": findings,
        })
        # If findings contain scalar scores, log them
        for h in r["headline"]:
            value = _to_float(h["value"])
            if value is not None:
                scores.append({
                    "name": f"m1_{r['name']}_{h['label'].lower().replace(' ', '_')}",
                    "value": value,
                    "comment": f"{r['display_name']} - {h['label']}",
                    "span_id": probe_span_id,
                })

    # M2 Span
    m2_span_id = f"{trace_id}_m2"
    spans.append({
        "id": m2_span_id,
        "name": "M2: Screening & Confirmatory Signals (EDA)",
        "type": "span",
        "metadata": {
            "stage": "M2",
            "severity": m2["severity"],
            "duration_sec": m2["duration"],
            "n_tests": len(m2["stats"]),
            "n_rejected": sum(1 for s in m2["stats"] if s["reject"]),
        },
        "input": {"n_signals_screened": len(m2["stats"])},
        "output": {
            "conclusion": m2["conclusion"],
            "significant_signals": [
                s.get("config", {}).get("signal") or s.get("tool")
                for s in m2["stats"] if s["reject"]
            ],
        },
    })
    for s in m2["stats"]:
        sig_name = s.get("config", {}).get("signal") or s.get("tool") or "stat_test"
        if s.get("p_value") is not None:
            scores.append({
                "name": f"m2_fdr_p_{sig_name}",
                "value": float(s["p_value"]),
                "comment": f"FDR p-value (reject={s['reject']})",
                "span_id": m2_span_id,
            })

    # M3 Span & LLM Diagnostician Generation
    m3_span_id = f"{trace_id}_m3"
    spans.append({
        "id": m3_span_id,
        "name": "M3: Root-Cause Diagnosis (AI Doctor)",
        "type": "span",
        "metadata": {
            "stage": "M3",
            "duration_sec": m3["duration"],
            "n_hypotheses": len(m3["hypotheses"]),
        },
        "input": {"screened_anomalies": m2["conclusion"]},
        "output": {"hypotheses": m3["hypotheses"]},
    })
    for idx, h in enumerate(m3["hypotheses"], 1):
        generations.append({
            "id": f"gen_m3_hyp_{idx}",
            "trace_id": trace_id,
            "span_id": m3_span_id,
            "name": f"AI Doctor Diagnosis #{idx}",
            "model": "AI Diagnostician Agent",
            "prompt": f"Analyze screened signals: {m2['conclusion']}",
            "completion": json.dumps(h, indent=2, ensure_ascii=False),
            "metadata": {"failure_mode": h.get("failure_mode")},
        })

    # M4 Span
    if m4["ran"]:
        spans.append({
            "id": f"{trace_id}_m4",
            "name": "M4: Independent Blind Adjudication",
            "type": "span",
            "metadata": {
                "stage": "M4",
                "results_count": len(m4["results"]),
            },
            "input": {"hypotheses_tested": [h.get("statement") for h in m3["hypotheses"]]},
            "output": {"event": m4["event"], "results": m4["results"]},
        })

    # M5 Surgery Span
    if m5_s["ran"]:
        spans.append({
            "id": f"{trace_id}_m5_surgery",
            "name": "M5-SURGERY: Causal Mechanism Interventions",
            "type": "span",
            "metadata": {"stage": "M5-Surgery"},
            "output": {"surgeries": m5_s["surgeries"]},
        })

    # M5 Fix Span & Generations
    if m5_f["ran"]:
        m5_span_id = f"{trace_id}_m5_fix"
        spans.append({
            "id": m5_span_id,
            "name": "M5-FIX: Targeted Repair & Confirmation",
            "type": "span",
            "metadata": {
                "stage": "M5-Fix",
                "fixed": m5_f["fixed"],
                "n_candidates_screened": len(m5_f["selection"]),
            },
            "input": {"candidates": [s.get("name") for s in m5_f["selection"]]},
            "output": {
                # Fix events written before PR #88 carry ``best`` as the candidate
                # NAME, later ones as the candidate record: accept both.
                "best_candidate": (
                    (m5_f.get("best") or {}).get("name")
                    if isinstance(m5_f.get("best"), dict) else (m5_f.get("best") or None)
                ) or (m5_f.get("confirm") or {}).get("name"),
                "confirm": m5_f.get("confirm"),
            },
        })
        if m5_f.get("prompt_template"):
            generations.append({
                "id": "gen_m5_patch",
                "trace_id": trace_id,
                "span_id": m5_span_id,
                "name": "Winning Repair Patch",
                "model": "Repair Search Agent",
                "prompt": "Synthesize targeted prompt patch based on diagnosed mechanism",
                "completion": m5_f["prompt_template"],
                "metadata": {"candidate_name": (m5_f.get("confirm") or {}).get("name")},
            })

    # Global Key Evaluation Scores
    cfm = m5_f.get("confirm") or {}
    if cfm.get("n_baseline_correct") is not None and cfm.get("n_pairs"):
        scores.append({
            "name": "baseline_accuracy",
            "value": cfm["n_baseline_correct"] / cfm["n_pairs"],
            "comment": f"Baseline Correct: {cfm['n_baseline_correct']}/{cfm['n_pairs']}",
        })
    if cfm.get("effect") is not None:
        scores.append({
            "name": "repair_net_gain",
            "value": float(cfm["effect"]),
            "comment": f"Net accuracy shift: {float(cfm['effect']) * 100:+.2f}%",
        })
    if cfm.get("e_value") is not None:
        scores.append({
            "name": "evidence_strength_e_value",
            "value": float(cfm["e_value"]),
            "comment": "Multiplicity-corrected certainty",
        })

    bundle = {
        "trace": trace,
        "spans": spans,
        "generations": generations,
        "scores": scores,
    }

    if out_json:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"[✓] Exported Langfuse bundle to: {out_p}")

    return bundle


def backfill_run_to_langfuse(run_dir: str | Path, *, dry_run: bool = False) -> dict[str, int | str]:
    """Queue an existing JSONL run for the same reliable Langfuse pipeline.

    Existing ``event_seq`` values are preserved; older logs without one receive
    their line order.  Re-running the command is safe because envelope IDs are
    deterministic and the outbox primary key de-duplicates them.
    """
    root = Path(run_dir)
    log_path = root / "run_log.jsonl"
    if not log_path.exists() and (root / "logs" / "run_log.jsonl").exists():
        root = root / "logs"
        log_path = root / "run_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"No run_log.jsonl found under {run_dir}")

    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        return {"trace_id": "", "events": 0, "pending": 0, "published": 0}

    trace_id = str(next((r.get("trace_id") for r in records if r.get("trace_id")), uuid.uuid4()))
    if dry_run:
        return {"trace_id": trace_id, "events": len(records), "pending": len(records), "published": 0}

    tracer = DiagnosticTracer(run_dir=root)
    tracer.trace_id = trace_id
    run_start = next((r for r in records if r.get("event") == "run_start"), {})
    tracer.start_trace(
        model=str(run_start.get("model") or "Target Model"),
        benchmark=str(run_start.get("benchmark_name") or "Benchmark"),
        n_cases=int(run_start.get("n_cases") or 0),
        metadata=run_start,
    )
    for fallback_seq, record in enumerate(records, start=1):
        tracer.record_event(record, event_seq=int(record.get("event_seq") or fallback_seq))
    pending_before = tracer.outbox.pending_count()
    tracer.flush()
    pending_after = tracer.outbox.pending_count()
    tracer.end_trace({"backfilled_events": len(records)})
    tracer.flush()
    return {
        "trace_id": trace_id,
        "events": len(records),
        "pending": pending_after,
        "published": max(0, pending_before - pending_after),
    }


def sync_to_langfuse_live(run_dir: str | Path) -> bool:
    """If langfuse SDK is installed and credentials exist, push live to Langfuse.

    Uses the run's own trace id (from langfuse_trace.json / run_log.jsonl) so a
    batch re-sync lands on the SAME trace the live mirroring wrote to, rather
    than creating a duplicate.
    """
    try:
        from langfuse import Langfuse
    except ImportError:
        print("[!] `langfuse` package is not installed. Install with `pip install langfuse` to enable live sync.")
        return False

    bundle = export_to_langfuse_bundle(run_dir)
    trace_info = bundle["trace"]
    lf_trace_id = str(trace_info["id"]).replace("-", "")

    def _tags(client: "Langfuse", tags: list) -> None:
        try:
            fn = getattr(client, "_create_trace_tags_via_ingestion", None)
            if fn is not None:
                fn(trace_id=lf_trace_id, tags=[str(t) for t in tags if t])
        except Exception:
            pass

    try:
        langfuse = Langfuse()
        root = langfuse.start_observation(
            trace_context={"trace_id": lf_trace_id},
            name=trace_info["name"],
            as_type="chain",
            input=trace_info.get("metadata"),
            metadata=trace_info.get("metadata"),
        )
        _tags(langfuse, trace_info.get("tags") or [])

        obs_map: dict[str, Any] = {}
        for span in bundle["spans"]:
            parent = obs_map.get(span.get("parent_id")) or root
            try:
                obs = parent.start_observation(
                    name=span["name"],
                    input=span.get("input"),
                    output=span.get("output"),
                    metadata=span.get("metadata"),
                )
            except AttributeError:
                obs = langfuse.start_observation(
                    trace_context={"trace_id": lf_trace_id},
                    name=span["name"],
                    input=span.get("input"),
                    output=span.get("output"),
                    metadata=span.get("metadata"),
                )
            obs_map[span["id"]] = obs
            obs.end()

        for gen in bundle.get("generations", []):
            parent = obs_map.get(gen.get("span_id")) or root
            try:
                g = parent.start_observation(
                    name=gen["name"],
                    as_type="generation",
                    model=gen.get("model"),
                    input=gen.get("prompt"),
                    output=gen.get("completion"),
                    metadata=gen.get("metadata"),
                )
            except AttributeError:
                g = langfuse.start_observation(
                    trace_context={"trace_id": lf_trace_id},
                    name=gen["name"],
                    as_type="generation",
                    model=gen.get("model"),
                    input=gen.get("prompt"),
                    output=gen.get("completion"),
                    metadata=gen.get("metadata"),
                )
            g.end()

        for score in bundle.get("scores", []):
            obs = obs_map.get(score.get("span_id"))
            if obs is not None:
                obs.score(
                    name=score["name"],
                    value=score["value"],
                    comment=score.get("comment", ""),
                )
            else:
                langfuse.create_score(
                    trace_id=lf_trace_id,
                    name=score["name"],
                    value=score["value"],
                    comment=score.get("comment", ""),
                )

        root.update(output={"spans": len(bundle["spans"]), "generations": len(bundle.get("generations", []))})
        root.end()
        langfuse.flush()
        print(f"[✓] Successfully pushed trace to Langfuse dashboard: {trace_info['name']}")
        return True
    except Exception as e:
        print(f"[!] Langfuse sync failed: {e}")
        return False
