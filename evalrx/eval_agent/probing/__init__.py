"""Triage-first probing controller (``--m1-policy triage_active``).

The default M1 lets the decision judge pick a batch of analyzers, which
``ProbeAgent`` may then backfill, cap, or extend with generated probes. This
package is the opt-in alternative from ``evalsmith/probing/probing_design_mix.md``:
a host-side controller that runs one explicit action at a time, on exactly the
cases and configuration it planned, and records why each action ran, what it
cost, and whether its measurement can be trusted.

MVP-0 is a fixed plan with no judge and no M2-M5:

  partition.py  the same EXPLORE/CONFIRM split the diagnosis loop uses
  sampler.py    matched, nested FAIL/PASS sample (S8 ⊆ S16 ⊆ S32 ...)
  actions.py    ProbeAction / Observation and their content identities
  metering.py   per-action call counts, model seconds, GPU peak memory
  executor.py   strict execution: capability gate, typed outcome, no backfill
  ledger.py     append-only selection_trace.jsonl / cost_ledger.jsonl + resume
  triage.py     H/N/X route labels traced back to case, field and action
  controller.py the MVP-0 plan and run_triage_mvp0()

Nothing here is imported by the default ``judge`` path.
"""

from __future__ import annotations

from evalrx.eval_agent.probing.actions import Observation, ProbeAction
from evalrx.eval_agent.probing.controller import MVP0_PLAN, POLICY_VERSION, run_triage_mvp0
from evalrx.eval_agent.probing.partition import explore_confirm_split
from evalrx.eval_agent.probing.sampler import SampleDraw, matched_sample

__all__ = [
    "MVP0_PLAN",
    "POLICY_VERSION",
    "Observation",
    "ProbeAction",
    "SampleDraw",
    "explore_confirm_split",
    "matched_sample",
    "run_triage_mvp0",
]
