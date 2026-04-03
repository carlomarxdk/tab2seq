"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from tab2seq.config import (
    Config,
    LoaderConfig,
    ProcessorConfig,
    TokenizerConfig,
)
from tab2seq.tokenization.config import VocabularyConfig


def test_tokenizer_config_defaults():
    """Test default tokenizer configuration."""
    config = TokenizerConfig()
    assert config.vocab_size == 10000
    assert config.pad_token == "[PAD]"
    assert config.unk_token == "[UNK]"
    assert config.cls_token == "[CLS]"
    assert config.sep_token == "[SEP]"
    assert config.mask_token == "[MASK]"
    assert isinstance(config.vocabulary, VocabularyConfig)
    assert config.vocabulary.max_vocab_size == 50000
    assert config.vocabulary.min_token_count == 1


def test_loader_config_defaults():
    """Test default loader configuration."""
    config = LoaderConfig()
    assert config.chunk_size == 10000
    assert config.entity_id_column == "entity_id"
    assert config.timestamp_column == "timestamp"
    assert config.event_columns == []


def test_processor_config_defaults():
    """Test default processor configuration."""
    config = ProcessorConfig()
    assert config.batch_size == 32
    assert config.max_sequence_length == 512
    assert config.n_jobs == 1


def test_config_defaults():
    """Test default main configuration."""
    config = Config()
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert isinstance(config.loader, LoaderConfig)
    assert isinstance(config.processor, ProcessorConfig)
    assert config.output_dir == Path("./output")


def test_config_yaml_roundtrip():
    """Test saving and loading configuration from YAML."""
    config = Config()
    config.tokenizer.vocab_size = 5000
    config.tokenizer.vocabulary.max_vocab_size = 20000
    config.loader.chunk_size = 5000
    config.processor.batch_size = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        config.to_yaml(yaml_path)

        loaded_config = Config.from_yaml(yaml_path)
        assert loaded_config.tokenizer.vocab_size == 5000
        assert loaded_config.tokenizer.vocabulary.max_vocab_size == 20000
        assert loaded_config.loader.chunk_size == 5000
        assert loaded_config.processor.batch_size == 64


def test_config_from_yaml_not_found():
    """Test loading configuration from non-existent file."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent.yaml")


def test_config_output_dir_path_conversion():
    """Test output_dir path conversion."""
    config = Config(output_dir="/tmp/output")
    assert isinstance(config.output_dir, Path)
    assert config.output_dir == Path("/tmp/output")
