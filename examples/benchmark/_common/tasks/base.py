"""Compatibility alias for evalrx.benchmark.tasks.base."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.base')
