"""Compatibility alias for evalrx.benchmark.tasks.mmsu."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.mmsu')
