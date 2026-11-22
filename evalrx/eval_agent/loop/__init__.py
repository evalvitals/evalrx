"""The diagnosis loop and who picks its next step.

  diagnose.py  VLDiagnoseLoop — configuration and the run phases (run, run_analysis,
               run_confirm, run_m5, run_fix)
  stages.py    Stages — runs and records M1 probe, explore, M2, M3, M4; StageMemory
  partition.py split_cases — the EXPLORE / CONFIRM / TEST partition
  state.py     Policy (the next-step contract) and RunState (what a run carries)
  fixed.py     FixedOrderPolicy — M1→M4 cycles, the default
  judge/       JudgePolicy — a judge picks each next stage
  report.py    AutoDiagnoseReport and its stopped_by values
  fix.py       the post-loop fix stage: targets, tier ladder, EXPLORE→CONFIRM
  metadata.py  run provenance and logger wiring shared by the loop and policies
"""
