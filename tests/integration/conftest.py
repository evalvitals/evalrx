"""Session-scoped fixtures for GPU integration tests.

Load each model once per test session and share the handle across all
integration test files to avoid redundant weight loads (~2s from cache each).
"""

from __future__ import annotations

import pytest

from evalrx.core.capability import Capability
from evalrx.models import load


@pytest.fixture(scope="session")
def qwen_model():
    return load("qwen2.5-7b-instruct", want={Capability.ATTENTION, Capability.HIDDEN_STATES})
