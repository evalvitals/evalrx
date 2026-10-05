"""Compatibility alias for evalrx.benchmark.tasks._audio."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks._audio')
