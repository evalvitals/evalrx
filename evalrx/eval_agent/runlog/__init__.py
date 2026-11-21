"""A run's output directory and event log (see README.md in this folder).

  context.py          RunContext, Trial — the one owner of a run's directory
  logger.py           RunLogger — run.json + per-stage M<n>/log.json
  entries.py          what each logged event records
  files.py            atomic writes, inlined workspaces, saved .npy artifacts
  tracing.py          the run log mirrored into Langfuse
  schema.py           the published run_log JSON Schema (run_log.schema.json)
  narration.py        the live terminal narration of a run
  instrumentation.py  InstrumentedModel — logs each call an analyzer makes to the model
"""
