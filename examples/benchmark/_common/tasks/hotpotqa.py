"""Compatibility alias for evalrx.benchmark.tasks.hotpotqa."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.hotpotqa')
