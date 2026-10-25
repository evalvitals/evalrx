"""Triage-first probing controller (``--m1-policy triage_active``).

The default M1 lets the decision judge pick a batch of analyzers, which
``ProbeAgent`` may then backfill, cap, or extend with generated probes. This
package is the opt-in alternative from ``evalsmith/probing/probing_design_mix.md``:
a host-side controller that runs one explicit action at a time, on exactly the
cases and configuration it planned, and records why each action ran, what it
cost, and whether its measurement can be trusted.

A run draws a matched, nested sample of EXPLORE, runs a fixed battery on it, and
then decides one action at a time between expanding the sample and the routes'
follow-ups: by fixed rules, by value (optionally with competing explanations a
judge proposes, which give the value its D), or by the random and cheapest-first
controls. The issues a report supports can be frozen and validated once on cases
the controller never saw.

  partition.py  the EXPLORE/CONFIRM split the diagnosis loop uses, or one by media content
  sampler.py    matched, nested FAIL/PASS sample (S8 ⊆ S16 ⊆ S32 ...)
  actions.py    PlanStep / ProbeAction / Observation and their content identities
  metering.py   per-action call counts, model seconds, GPU peak memory, API usage
  executor.py   strict execution: capability gate, typed outcome, no backfill
  ledger.py     append-only selection_trace.jsonl / cost_ledger.jsonl + resume
  evidence.py   one step's observations over a growing sample (per-case reuse)
  triage.py     H/N/U/S/X route labels traced back to case, field and action
  rules.py      route layers, follow-up choice, and the expansion rule
  scoring.py    the scored scheduler's value of a candidate, and the controls' pick
  explanations.py competing explanations, their predictions and D
  controller.py the battery, the schedulers and run_triage()
  validation.py the data roles, frozen issues and their one validation
  compare.py    two arms' reports side by side (also a CLI)

Nothing here is imported by the default ``judge`` path.
"""

from __future__ import annotations

from evalrx.eval_agent.probing.actions import Observation, PlanStep, ProbeAction
from evalrx.eval_agent.probing.controller import BATTERY, POLICY_VERSION, run_triage
from evalrx.eval_agent.probing.partition import explore_confirm_split
from evalrx.eval_agent.probing.rules import LAYERS, RULES_VERSION
from evalrx.eval_agent.probing.sampler import SampleDraw, matched_sample

__all__ = [
    "BATTERY",
    "LAYERS",
    "POLICY_VERSION",
    "RULES_VERSION",
    "Observation",
    "PlanStep",
    "ProbeAction",
    "SampleDraw",
    "explore_confirm_split",
    "matched_sample",
    "run_triage",
]
