"""Hallucination diagnostics — the dominant MLLM failure mode (BB probes + WB attention)."""

from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer, chair_score
from evalvitals.analyzers.hallucination.opera import OPERAAnalyzer
from evalvitals.analyzers.hallucination.pope import POPEAnalyzer
from evalvitals.analyzers.hallucination.selfcheck import SelfCheckConsistencyAnalyzer
from evalvitals.analyzers.hallucination.vcd import VCDAnalyzer

__all__ = ["POPEAnalyzer", "CHAIRAnalyzer", "chair_score", "OPERAAnalyzer", "VCDAnalyzer", "SelfCheckConsistencyAnalyzer"]
