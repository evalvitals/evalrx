"""The diagnosis loop and who picks its next step.

  diagnose.py  VLDiagnoseLoop — the stages, the confirm split, the run phases
  state.py     Policy (the next-step contract) and RunState (what a run carries)
  fixed.py     FixedOrderPolicy — M1→M4 cycles, the default
  judge/       JudgePolicy — a judge picks each next stage
  report.py    AutoDiagnoseReport and its stopped_by values
  fix.py       the post-loop fix stage: targets, tier ladder, EXPLORE→CONFIRM
  metadata.py  run provenance and logger wiring shared by the loop and policies
"""
