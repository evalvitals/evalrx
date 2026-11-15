"""Compatibility alias for evalrx.benchmark.models."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.models')
