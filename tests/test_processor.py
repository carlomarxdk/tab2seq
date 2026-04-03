"""Tests for batch processor module."""

import polars as pl
import pytest

from tab2seq.config import ProcessorConfig
from tab2seq.processor import BatchProcessor
from tab2seq import Tokenizer


@pytest.fixture
def tokenizer():
    """Create a tokenizer with sample vocabulary."""
    tok = Tokenizer()
    events = [
        pl.DataFrame(
            {
                "event_type": ["A", "B", "C"],
                "value": [10, 20, 30],
            }
        )
    ]
    tok.fit(events)
    return tok


@pytest.fixture
def sample_persons():
    """Create sample person data."""
    return [
        ("person1", pl.DataFrame({"event_type": ["A", "B"], "value": [10, 20]})),
        ("person2", pl.DataFrame({"event_type": ["C"], "value": [30]})),
        ("person3", pl.DataFrame({"event_type": ["A", "C"], "value": [15, 35]})),
    ]


def test_processor_initialization(tokenizer):
    """Test BatchProcessor initialization."""
    processor = BatchProcessor(tokenizer)
    assert processor.config.batch_size == 32
    assert processor.config.n_jobs == 1


def test_processor_with_config(tokenizer):
    """Test BatchProcessor with custom config."""
    config = ProcessorConfig(batch_size=16, n_jobs=2)
    processor = BatchProcessor(tokenizer, config)
    assert processor.config.batch_size == 16
    assert processor.config.n_jobs == 2


def test_process_entity(tokenizer, sample_entities):
    """Test processing a single entity."""
    processor = BatchProcessor(tokenizer)
    result = processor.process_entity(sample_entities[0])

    assert "entity_id" in result
    assert "token_ids" in result
    assert "length" in result
    assert result["entity_id"] == "entity1"
    assert isinstance(result["token_ids"], list)
    assert len(result["token_ids"]) == processor.config.max_sequence_length


def test_process_batch_sequential(tokenizer, sample_entities):
    """Test processing a batch sequentially."""
    config = ProcessorConfig(n_jobs=1)
    processor = BatchProcessor(tokenizer, config)
    results = processor.process_batch(sample_entities)
    assert len(results) == 3
    assert all("entity_id" in r for r in results)
    assert all("token_ids" in r for r in results)


def test_process_batch_parallel(tokenizer, sample_persons):
    """Test processing a batch in parallel."""
    config = ProcessorConfig(n_jobs=2)
    processor = BatchProcessor(tokenizer, config)
    results = processor.process_batch(sample_persons)

    assert len(results) == 3
    entity_ids = [r["entity_id"] for r in results]
    assert set(entity_ids) == {"person1", "person2", "person3"}


def test_process_stream(tokenizer, sample_entities):
    """Test processing a stream of entities."""
    config = ProcessorConfig(batch_size=2)
    processor = BatchProcessor(tokenizer, config)

    batches = list(processor.process_stream(iter(sample_entities)))
    # Should have 2 batches: [2 entities, 1 entity]
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_process_stream_exact_batch(tokenizer, sample_persons):
    """Test processing stream with exact batch size."""
    config = ProcessorConfig(batch_size=3)
    processor = BatchProcessor(tokenizer, config)

    batches = list(processor.process_stream(iter(sample_persons)))

    # Should have exactly 1 batch
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_processor_close(tokenizer):
    """Test closing processor."""
    processor = BatchProcessor(tokenizer)
    processor.close()  # Should not raise
