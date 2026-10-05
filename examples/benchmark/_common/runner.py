"""Compatibility alias for evalrx.benchmark.runner."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.runner')
