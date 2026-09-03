"""Hallucination diagnostics — the dominant MLLM failure mode (BB probes + WB attention)."""

from evalrx.analyzers.hallucination.chair import CHAIRAnalyzer, chair_score
from evalrx.analyzers.hallucination.opera import OPERAAnalyzer
from evalrx.analyzers.hallucination.pope import POPEAnalyzer
from evalrx.analyzers.hallucination.selfcheck import SelfCheckConsistencyAnalyzer
from evalrx.analyzers.hallucination.vcd import VCDAnalyzer

__all__ = ["POPEAnalyzer", "CHAIRAnalyzer", "chair_score", "OPERAAnalyzer", "VCDAnalyzer", "SelfCheckConsistencyAnalyzer"]
