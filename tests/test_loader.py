"""Tests for data loader module."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from tab2seq.config import LoaderConfig
from tab2seq.loader import DataLoader


@pytest.fixture
def sample_data():
    """Create sample event data."""
    return pl.DataFrame(
        {
            "entity_id": [1, 1, 2, 2, 3, 3, 3],
            "timestamp": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "event_type": ["A", "B", "A", "C", "B", "A", "C"],
            "value": [10, 20, 15, 25, 30, 35, 40],
        }
    )


@pytest.fixture
def csv_file(sample_data):
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.write_csv(f.name)
        yield Path(f.name)
    Path(f.name).unlink()


def test_loader_initialization():
    """Test DataLoader initialization."""
    loader = DataLoader()
    assert loader.config.chunk_size == 10000


def test_loader_with_config():
    """Test DataLoader with custom config."""
    config = LoaderConfig(chunk_size=100)
    loader = DataLoader(config)
    assert loader.config.chunk_size == 100


def test_load_chunks_csv(csv_file):
    """Test loading CSV in chunks."""
    config = LoaderConfig(chunk_size=3)
    loader = DataLoader(config)

    chunks = list(loader.load_chunks(csv_file))
    assert len(chunks) >= 2  # Should have multiple chunks


def test_load_chunks_not_found():
    """Test loading from non-existent file."""
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        list(loader.load_chunks("nonexistent.csv"))


def test_group_by_entity(sample_data):
    """Test grouping events by entity."""
    loader = DataLoader()
    chunks = [sample_data]
    entities = list(loader.group_by_entity(iter(chunks)))
    assert len(entities) == 3
    entity_ids = [p[0] for p in entities]
    assert set(entity_ids) == {1, 2, 3}


def test_group_by_entity_missing_column():
    """Test grouping with missing entity_id column."""
    df = pl.DataFrame({"event": ["A", "B"]})
    loader = DataLoader()

    with pytest.raises(ValueError, match="Entity ID column"):
        list(loader.group_by_entity(iter([df])))


def test_load_entities(csv_file):
    """Test loading and grouping entities."""
    loader = DataLoader()
    entities = list(loader.load_entities(csv_file))

    assert len(entities) == 3
    entity_ids = [p[0] for p in entities]
    assert set(entity_ids) == {1, 2, 3}

    # Check entity 3 has 3 events
    entity_3_events = [p[1] for p in entities if p[0] == 3][0]
    assert len(entity_3_events) == 3
