"""Compatibility entry point for evalrx run."""
import sys
from evalrx.benchmark import run as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
