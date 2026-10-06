"""Compatibility alias for evalrx.benchmark.scoring."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.scoring')
