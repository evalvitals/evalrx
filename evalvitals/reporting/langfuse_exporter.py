"""Langfuse observability & exporter facade for backward compatibility."""

from __future__ import annotations

from evalvitals.observability.tracer import (
    DiagnosticTracer,
    export_to_langfuse_bundle,
    sync_to_langfuse_live,
)

__all__ = [
    "DiagnosticTracer",
    "export_to_langfuse_bundle",
    "sync_to_langfuse_live",
]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Export an EvalVitals run into a Langfuse ingestion bundle.")
    ap.add_argument("run_dir", nargs="?", default="outputs", help="Run directory.")
    ap.add_argument("--out", "-o", default=None, help="Output JSON path.")
    ap.add_argument("--sync", action="store_true", help="Push live to Langfuse API.")
    args = ap.parse_args()

    if args.sync:
        sync_to_langfuse_live(args.run_dir)
    else:
        out_p = args.out or (Path(args.run_dir) / "langfuse_trace.json")
        export_to_langfuse_bundle(args.run_dir, out_p)


if __name__ == "__main__":
    main()
