"""Built-in datasets and synthetic data generation."""

from tab2seq.datasets.builder import EventDataset
from tab2seq.datasets.config import EventDatasetConfig, RelativeDateRule
from tab2seq.datasets.synthetic import (
    generate_synthetic_collections,
    generate_synthetic_data,
)

__all__ = [
    "EventDataset",
    "EventDatasetConfig",
    "RelativeDateRule",
    "generate_synthetic_collections",
    "generate_synthetic_data",
]
