"""Project-level configuration models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from tab2seq.tokenization.config import TokenizerConfig, VocabularyConfig


class LoaderConfig(BaseModel):
    """Data loading configuration."""

    chunk_size: int = Field(default=10000, gt=0)
    entity_id_column: str = "entity_id"
    timestamp_column: str = "timestamp"
    event_columns: list[str] = Field(default_factory=list)


class ProcessorConfig(BaseModel):
    """Batch processing configuration."""

    batch_size: int = Field(default=32, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)
    n_jobs: int = Field(default=1, gt=0)


class Config(BaseModel):
    """Top-level project configuration."""

    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    output_dir: Path = Path("./output")

    @field_validator("output_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path) -> Path:
        return Path(v)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to yaml."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from yaml."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
