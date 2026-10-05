"""Compatibility alias for evalrx.benchmark.tasks.af_reasoning_mcq."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.af_reasoning_mcq')
