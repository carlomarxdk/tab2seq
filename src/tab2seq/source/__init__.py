"""Data source configuration, access, and collection."""

from tab2seq.source.collection import SourceCollection
from tab2seq.source.config import SourceConfig
from tab2seq.source.core import SchemaError, Source

__all__ = [
    "Source",
    "SourceCollection",
    "SourceConfig",
    "SchemaError",
]
