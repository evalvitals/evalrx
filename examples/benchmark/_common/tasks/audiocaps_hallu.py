"""Compatibility alias for evalrx.benchmark.tasks.audiocaps_hallu."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('evalrx.benchmark.tasks.audiocaps_hallu')
