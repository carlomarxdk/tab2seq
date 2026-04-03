"""Shared pytest fixtures for test suite compatibility."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def sample_entities() -> list[tuple[str, pl.DataFrame]]:
    """Entity-aligned sample event frames used by processor tests."""
    return [
        ("entity1", pl.DataFrame({"event_type": ["A", "B"], "value": [10, 20]})),
        ("entity2", pl.DataFrame({"event_type": ["C"], "value": [30]})),
        ("entity3", pl.DataFrame({"event_type": ["A", "C"], "value": [15, 35]})),
    ]
