"""Compatibility alias for evalrx.benchmark.tasks.chair_words."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.chair_words')
