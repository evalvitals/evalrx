"""Stage implementations for the M1–M5 diagnostic pipeline.

Each module is a self-contained stage that can be injected into either loop:

  AutoDiagnoseLoop  uses M1 → M2 → M3 → M4
  VLDiagnoseLoop    uses M1 → M2 → M3 → M5 (inner), M4 called post-loop

M1 — Probe (analyzer selection + execution)
  probe.py        StrategyProbe   detect model kind; rank analyzers by failure-mode hints
  probe_agent.py  ProbeAgent      execute ranked analyzers; route to Docker if needed;
                                  protocol-guided via LLM analyzer selection
  protocol.py     ExperimentProtocol   user's NL description of what to investigate;
                  ProbingSchema        records which analyzers ran and why (M2/M5 trace)

Data — Case discovery / labeling
  case_discovery.py  CaseDiscoveryAgent  run candidate prompts through the model,
                                         store observed outputs, and label PASS/FAIL

M2 — Analysis (statistical interpretation)
  Lives in ``evalrx.analysis`` (not this package) so it is usable standalone:
  analysis_module.py  AnalysisModule       threshold rules + VLM derived metrics → AnalysisReport
  stats_agent.py       StatsAnalysisAgent   protocol-aware extension of AnalysisModule;
                                            adds LLM-written conclusion + evidence chain
                       StatsAnalysisReport  AnalysisReport subclass — adds conclusion,
                                            evidence_chain, qualitative_findings, protocol
  stats_tool_agent.py  StatsToolAgent       selects/runs deterministic exploratory
                                            stats tools and visualization specs
  stats_tools.py       STATS_TOOL_CATALOG   signal/label association, McNemar+e-value,
                                            Friedman, single-rate e-value, rank corr
  stats_tool_generator.py StatsToolGenerator  LLM/CLI writes a new stats script when
                                            no catalog tool fits

M3 — Diagnosis (hypothesis generation)
  diagnosis.py    DiagnosisAgent   LLM judge reads AnalysisReport → Hypothesis list;
                                   falls back to threshold-derived hypotheses on NO_ISSUE

M4 — Surgery (fix proposal + execution)
  surgery.py           SurgeryAgent       four strategies: verify_fn → analyzer_params →
                                          ExperimentWriter → label correlation
  experiment_writer.py ExperimentWriter   multi-phase LLM/CLI agent writes + runs
                                          diagnostic Python projects in a sandbox

M5 — Hypothesis testing (statistical + protocol consistency gate)
  hypothesis_tester.py HypothesisTester   per-case signal extraction → fail-rate comparison
                                          → optional LLM protocol consistency check;
                                          stopping_criteria_met() returns True when at
                                          least one hypothesis is SUPPORTED + consistent

Cross-stage shared types (top level, not in this package):
  hypothesis.py   Hypothesis, HypothesisStatus   used by M3, M4, M5

Shared CLI-agent runtime (evalrx.agent_runtime, not in this package):
  codegen/        CodegenRunner   shared CLI code-generation orchestration
  providers/      CLI coding-provider adapters (agy/codex/claude/etc.)
  judges/         CLI-backed judge model wrappers
"""
