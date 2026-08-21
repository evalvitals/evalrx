"""Claim/evidence diagnostic reporting."""

from evalvitals.reporting.compiler import compile_diagnostic_report
from evalvitals.reporting.html_report import build_html_report
from evalvitals.reporting.model import Claim, DiagnosticReport, Evidence, ReportStep
from evalvitals.reporting.stages import STAGE_SPECS, StageSpec, stage_specs_as_dicts

__all__ = [
    "Claim",
    "DiagnosticReport",
    "Evidence",
    "ReportStep",
    "STAGE_SPECS",
    "StageSpec",
    "build_html_report",
    "compile_diagnostic_report",
    "stage_specs_as_dicts",
]
