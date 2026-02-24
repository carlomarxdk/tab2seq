"""Collection of data sources with cross-source operations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from tab2seq.source.config import SourceConfig
from tab2seq.source.core import Source

logger = logging.getLogger(__name__)


class SourceCollection:
    """Container for multiple :class:`Source` objects.

    Provides dict-like access by source name and cross-source
    operations such as collecting all unique entity IDs.

    Parameters:
        sources: List of :class:`Source` instances.

    Example::

        collection = SourceCollection.from_yaml("config.yaml")

        # dict-like access
        health = collection["health"]

        # iteration
        for source in collection:
            print(source.name)

        # cross-source entity IDs
        all_ids = collection.get_all_entity_ids()
    """

    def __init__(self, sources: list[Source]) -> None:
        """Initialize the collection with a list of sources.

        Args:
            sources: List of :class:`Source` instances.
        """
        self._sources: dict[str, Source] = {s.name: s for s in sources}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(cls, configs: list[SourceConfig]) -> SourceCollection:
        """Create a collection from a list of config objects.

        Args:
            configs (list[SourceConfig]): Source configurations.

        Returns:
            A new :class:`SourceCollection`.
        """
        return cls([Source(cfg) for cfg in configs])

    @classmethod
    def from_yaml(cls, path: str | Path) -> SourceCollection:
        """Load sources defined under a ``sources`` key in a YAML file.

        Expected YAML structure::

            sources:
              - name: health
                filepath: data/health.parquet
                entity_id_col: pid
                timestamp_col: date
                categorical_cols: [diagnosis, department]
                continuous_cols: [cost]
              - name: income
                ...

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A new :class:`SourceCollection`.
        """
        path = Path(path)
        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        configs = [SourceConfig(**entry) for entry in raw["sources"]]
        logger.info("Loaded %d source(s) from %s", len(configs), path)
        return cls.from_configs(configs)

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Source:
        try:
            return self._sources[name]
        except KeyError:
            available = ", ".join(self._sources)
            msg = f"Source '{name}' not found. Available: {available}"
            raise KeyError(msg) from None

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self) -> Iterator[Source]:
        yield from self._sources.values()

    def __repr__(self) -> str:
        names = list(self._sources)
        return f"SourceCollection(sources={names})"

    @property
    def names(self) -> list[str]:
        """Names of all registered sources.

        Returns:
            list[str]: List of source names.
        """
        return list(self._sources)

    @property
    def sources(self) -> dict[str, Source]:
        """Dictionary of all sources in the collection.
        Returns:
            dict[str, Source]: Mapping of source names to Source objects.
        """
        return self._sources

    # ------------------------------------------------------------------
    # Cross-source operations
    # ------------------------------------------------------------------

    def get_all_entity_ids(self) -> set[str]:
        """Collect all unique entity IDs across every source.

        Returns:
            set[str]: Union of entity IDs from all sources.
        """
        all_ids: set[str] = set()
        for source in self:
            all_ids |= source.get_entity_ids()
        logger.info("Total unique entities across all sources: %d", len(all_ids))
        return all_ids

    def count_entities_per_source(self) -> dict[str, int]:
        """Count unique entities in each source.

        Returns:
            dict[str, int]: Dictionary mapping source names to their unique entity counts.
        """
        counts = {}
        for source in self:
            ids = source.get_entity_ids()
            counts[source.name] = len(ids)
            logger.info("Source '%s': %d unique entities", source.name, len(ids))
        return counts
