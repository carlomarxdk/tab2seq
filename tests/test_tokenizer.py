"""Tests for tokenizer module."""

import polars as pl
import pytest

from tab2seq.config import TokenizerConfig
from tab2seq import Tokenizer


@pytest.fixture
def sample_events():
    """Create sample event dataframes."""
    return [
        pl.DataFrame(
            {
                "event_type": ["A", "B", "A"],
                "value": [10, 20, 15],
            }
        ),
        pl.DataFrame(
            {
                "event_type": ["B", "C"],
                "value": [25, 30],
            }
        ),
    ]


def test_tokenizer_initialization():
    """Test Tokenizer initialization."""
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size >= 5  # At least special tokens


def test_tokenizer_special_tokens():
    """Test special tokens are in vocabulary."""
    config = TokenizerConfig()
    tokenizer = Tokenizer(config)

    assert config.pad_token in tokenizer.vocab
    assert config.unk_token in tokenizer.vocab
    assert config.cls_token in tokenizer.vocab
    assert config.sep_token in tokenizer.vocab
    assert config.mask_token in tokenizer.vocab


def test_fit_builds_vocabulary(sample_events):
    """Test that fit builds vocabulary from events."""
    tokenizer = Tokenizer()
    initial_size = tokenizer.vocab_size

    tokenizer.fit(sample_events)

    assert tokenizer.vocab_size > initial_size
    assert "event_type_A" in tokenizer.vocab
    assert "event_type_B" in tokenizer.vocab
    assert "value_10" in tokenizer.vocab


def test_fit_respects_vocab_size():
    """Test that fit respects maximum vocabulary size."""
    config = TokenizerConfig(vocab_size=10)  # Very small vocab
    tokenizer = Tokenizer(config)

    # Create many unique events
    events = [
        pl.DataFrame(
            {
                "event_type": [f"E{i}" for i in range(100)],
                "value": list(range(100)),
            }
        )
    ]

    tokenizer.fit(events)
    assert tokenizer.vocab_size <= 10


def test_encode_single_event():
    """Test encoding a single person's events."""
    tokenizer = Tokenizer()
    events = pl.DataFrame(
        {
            "event_type": ["A", "B"],
            "value": [10, 20],
        }
    )

    tokenizer.fit([events])
    token_ids = tokenizer.encode(events)

    # Should have CLS + tokens + SEP
    assert token_ids[0] == tokenizer.vocab[tokenizer.config.cls_token]
    assert token_ids[-1] == tokenizer.vocab[tokenizer.config.sep_token]
    assert len(token_ids) > 2


def test_encode_with_columns():
    """Test encoding with specific columns."""
    tokenizer = Tokenizer()
    events = pl.DataFrame(
        {
            "event_type": ["A", "B"],
            "value": [10, 20],
            "ignore": ["X", "Y"],
        }
    )

    tokenizer.fit([events], columns=["event_type", "value"])
    token_ids = tokenizer.encode(events, columns=["event_type", "value"])

    # Should not include 'ignore' column
    decoded = tokenizer.decode(token_ids)
    assert not any("ignore" in token for token in decoded)


def test_decode():
    """Test decoding token IDs back to tokens."""
    tokenizer = Tokenizer()
    events = pl.DataFrame({"event_type": ["A"]})

    tokenizer.fit([events])
    token_ids = tokenizer.encode(events)
    tokens = tokenizer.decode(token_ids)

    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)


def test_pad_sequence_truncate():
    """Test padding/truncation of sequences."""
    tokenizer = Tokenizer()
    token_ids = [1, 2, 3, 4, 5]

    # Truncate
    padded = tokenizer.pad_sequence(token_ids, max_length=3)
    assert len(padded) == 3
    assert padded[-1] == tokenizer.vocab[tokenizer.config.sep_token]


def test_pad_sequence_pad():
    """Test padding of short sequences."""
    tokenizer = Tokenizer()
    token_ids = [1, 2, 3]

    # Pad
    padded = tokenizer.pad_sequence(token_ids, max_length=5)
    assert len(padded) == 5
    assert padded[-2:] == [tokenizer.vocab[tokenizer.config.pad_token]] * 2


def test_vocab_size_property():
    """Test vocab_size property."""
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == len(tokenizer.vocab)


def test_fit_excludes_entity_id_by_default():
    """Tokenizer should not add entity_id tokens when columns are not provided."""
    tokenizer = Tokenizer()
    events = [
        pl.DataFrame(
            {
                "entity_id": ["e1", "e1", "e2"],
                "event_type": ["A", "B", "A"],
            }
        )
    ]

    tokenizer.fit(events)

    assert "entity_id_e1" not in tokenizer.vocab
    assert "entity_id_e2" not in tokenizer.vocab
    assert "event_type_A" in tokenizer.vocab


def test_encode_excludes_entity_id_by_default():
    """Tokenizer should not emit entity_id tokens when columns are not provided."""
    tokenizer = Tokenizer()
    events = pl.DataFrame(
        {
            "entity_id": ["e1", "e1"],
            "event_type": ["A", "B"],
        }
    )

    tokenizer.fit([events])
    token_ids = tokenizer.encode(events)
    decoded = tokenizer.decode(token_ids)

    assert not any(token.startswith("entity_id_") for token in decoded)


def test_explicit_columns_can_include_entity_id():
    """Explicit columns should still allow entity_id tokenization when requested."""
    tokenizer = Tokenizer()
    events = [
        pl.DataFrame(
            {
                "entity_id": ["e1", "e2"],
                "event_type": ["A", "B"],
            }
        )
    ]

    tokenizer.fit(events, columns=["entity_id", "event_type"])

    assert "entity_id_e1" in tokenizer.vocab
