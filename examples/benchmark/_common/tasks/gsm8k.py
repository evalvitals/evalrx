"""Compatibility alias for evalrx.benchmark.tasks.gsm8k."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.gsm8k')
