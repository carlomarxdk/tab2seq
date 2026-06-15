from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tab2seq.cohort.config import CohortConfig, EntityInclusionCriteria
from tab2seq.source import Source, SourceCollection


logger = logging.getLogger("Cohort")


class Cohort:
    """Cohort unites the `Source` objects to create a unified entity set and filter, split, and cache it for modeling.
    """
    
    def __init__(
        self,
        name: str,
        sources: Source | list[Source] | SourceCollection,
        inclusion_criteria: list[EntityInclusionCriteria] | None = None,
        cache_dir: str | Path = Path("data/cohorts/"),
        use_cache: bool = True,
    ) -> None:
        """Initialize a Cohort with given sources and configuration.
        Args:
            name: Unique name for this cohort, used in caching and metadata.
            sources: One or more `Source` objects or a `SourceCollection` defining the data sources for this cohort.
            inclusion_criteria: Optional list of `EntityInclusionCriteria` to filter entities.
            cache_dir: Base directory for caching cohort artifacts. If None, caching is disabled.
            use_cache: Whether to enable caching for this cohort. If False, no caching will occur
                even if `cache_dir` is provided.
        Raises:
            ValueError: If `name` is empty or contains only whitespace, or if `inclusion_criteria` contains invalid entries.
            KeyError: If `inclusion_criteria` references a source not in the collection.
            TypeError: If `sources` is not a valid type (Source, list[Source], or SourceCollection).
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string.")
        if name != name.strip():
            raise ValueError("'name' cannot have leading or trailing whitespace.")

        self._name = name

        if isinstance(sources, Source):
            self._collection = SourceCollection([sources])
        elif isinstance(sources, list):
            self._collection = SourceCollection(sources)
        elif isinstance(sources, SourceCollection):
            self._collection = sources
        else:
            raise TypeError(
                "'sources' must be a Source, list[Source], or SourceCollection."
            )

        self._criteria = inclusion_criteria or []

        base_cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_dir = base_cache_dir / self._name if base_cache_dir else None
        if self._cache_dir and use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._use_cache = True
        else:
            logger.warning(
                "Caching is disabled for this cohort. To enable caching, provide a valid 'cache_dir' and set 'use_cache=True'."
            )
            self._use_cache = False
            self._cache_dir = None

        self._entity_ids: set[str] = self._resolve_entity_ids()
        self._entities_table: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Cohort name."""
        return self._name

    @property
    def entity_ids(self) -> set[str]:
        """Set of entity IDs in this cohort."""
        return set(self._entity_ids)

    @property
    def entity_id_list(self) -> list[str]:
        """Deterministically sorted entity IDs in this cohort."""
        return sorted(self._entity_ids)

    @property
    def cache_dir(self) -> Path | None:
        """Directory for caching cohort splits and metadata, or None if caching is disabled."""
        return self._cache_dir

    @property
    def use_cache(self) -> bool:
        """Whether to use caching for cohort splits."""
        return self._use_cache

    @property
    def criteria(self) -> list[EntityInclusionCriteria] | None:
        """List of `EntityInclusionCriteria` used to define this cohort, or None."""
        return list(self._criteria)

    @property
    def entities_table(self) -> pl.DataFrame | None:
        """Cached entities table (entity_id + static columns), if already built."""
        return self._entities_table

    @property
    def collection(self) -> SourceCollection:
        """Underlying source collection used by this cohort."""
        return self._collection

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entity_ids)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entity_ids

    def __repr__(self) -> str:
        return (
            f"Cohort("
            f"name={self._name}, "
            f"sources={self._collection.names}, "
            f"n_entities={len(self)}, "
            f"cache_dir={self._cache_dir}"
            f")"
        )

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_entity_ids(self) -> set[str]:
        """Apply inclusion criteria and return the surviving entity ID set.

        Starts from the union of all entity IDs across the collection, then
        applies each ``EntityInclusionCriteria`` in sequence.  Only criteria
        with ``required=True`` have a filtering effect — non-required entries
        are silently skipped.

        Returns:
            Set of entity IDs satisfying all inclusion criteria.

        Raises:
            KeyError: If a criteria references a source not in the collection.
        """
        candidates = self._collection.get_all_entity_ids()
        logger.info(
            "Candidate pool: %d entities (union across all sources)", len(candidates)
        )

        if not self._criteria:
            logger.info("No inclusion criteria provided; keeping all candidates.")
            return candidates

        for criteria in self._criteria:
            if not criteria.required:
                continue

            if criteria.source_name not in self._collection:
                raise KeyError(
                    f"Inclusion criteria references unknown source "
                    f"'{criteria.source_name}'. Available: {self._collection.names}"
                )

            source = self._collection[criteria.source_name]
            id_col = source.config.id_col
            before = len(candidates)

            qualifying = (
                source.scan()
                .group_by(id_col)
                .agg(pl.len().alias("_n_events"))
                .filter(pl.col("_n_events") >= criteria.min_events)
                .pipe(
                    lambda lf: (
                        lf.filter(pl.col("_n_events") <= criteria.max_events)
                        if criteria.max_events is not None
                        else lf
                    )
                )
                .select(id_col)
                .collect()
                .get_column(id_col)
                .to_list()
            )

            candidates &= set(qualifying)

            logger.info(
                "After criteria for '%s' (min=%s, max=%s): %d → %d entities",
                criteria.source_name,
                criteria.min_events,
                criteria.max_events,
                before,
                len(candidates),
            )

        logger.info("Resolved cohort: %d entities", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Public filtering helpers
    # ------------------------------------------------------------------

    def filter_df(self, df: pl.DataFrame, entity_id_col: str = "entity_id") -> pl.DataFrame:
        """Filter a DataFrame to only cohort entities."""
        return df.filter(pl.col(entity_id_col).cast(pl.Utf8).is_in(self.entity_id_list))

    def filter_source(self, source: Source) -> pl.LazyFrame:
        """Filter a source scan to only cohort entities."""
        return source.scan().filter(
            pl.col(source.config.id_col).cast(pl.Utf8).is_in(self.entity_id_list)
        )

    # ------------------------------------------------------------------
    # Entities table
    # ------------------------------------------------------------------

    def build_entities_table(self, force_recompute: bool = False) -> pl.DataFrame:
        """Build or load entity_id plus static properties for this cohort."""
        entities_path = self._entities_table_path()
        metadata_path = self._entities_metadata_path()

        if (
            self.use_cache
            and not force_recompute
            and entities_path is not None
            and entities_path.exists()
        ):
            logger.info("Loading entities table from cache: %s", entities_path)
            self._entities_table = pl.read_parquet(entities_path)
            return self._entities_table

        entity_df = pl.DataFrame(
            {"entity_id": pl.Series(self.entity_id_list, dtype=pl.Utf8)}
        )

        for source in self._collection:
            static_slice = self._build_source_static_slice(source)
            if static_slice is not None:
                entity_df = entity_df.join(static_slice, on="entity_id", how="left")

        entity_df = entity_df.sort("entity_id")
        self._entities_table = entity_df

        if self.use_cache and entities_path is not None and metadata_path is not None:
            entities_path.parent.mkdir(parents=True, exist_ok=True)
            entity_df.write_parquet(entities_path)
            metadata = {
                "cohort_name": self.name,
                "n_entities": entity_df.height,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "criteria_hash": self._criteria_hash(),
                "source_config_hashes": {
                    source.name: self._stable_hash(source.config.model_dump_json())
                    for source in self._collection
                },
                "columns": entity_df.columns,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return entity_df

    # ------------------------------------------------------------------
    # Split building
    # ------------------------------------------------------------------

    def build_or_load_splits(
        self,
        split_config: CohortConfig | None = None,
        force_recompute: bool = False,
    ) -> pl.DataFrame:
        """Build or load cohort splits with full static context per entity."""
        cfg = split_config or CohortConfig(
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
        )
        split_path = self._split_table_path(cfg)
        metadata_path = self._split_metadata_path(cfg)

        if (
            self.use_cache
            and not force_recompute
            and split_path is not None
            and split_path.exists()
        ):
            logger.info("Loading split table from cache: %s", split_path)
            return pl.read_parquet(split_path)

        entities = self.build_entities_table(force_recompute=force_recompute)
        split_labels = self._assign_splits(entities, cfg)
        split_df = entities.with_columns(pl.Series(name="split", values=split_labels))

        if self.use_cache and split_path is not None and metadata_path is not None:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_df.write_parquet(split_path)
            split_counts = (
                split_df.group_by("split").len().rename({"len": "count"}).to_dicts()
            )
            metadata = {
                "cohort_name": self.name,
                "split_config": cfg.model_dump(exclude_none=False),
                "split_config_hash": cfg.config_hash(),
                "n_entities": split_df.height,
                "split_counts": split_counts,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return split_df

    # ------------------------------------------------------------------
    # Cache paths
    # ------------------------------------------------------------------

    def _entities_table_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "entities" / "entities_with_static.parquet"

    def _entities_metadata_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "entities" / "metadata.json"

    def _split_table_path(self, cfg: CohortConfig) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "splits" / cfg.config_hash() / "entities_split.parquet"

    def _split_metadata_path(self, cfg: CohortConfig) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "splits" / cfg.config_hash() / "metadata.json"

    def vocabulary_cache_dir(self, vocab_hash: str) -> Path | None:
        """Return cache directory for a specific vocabulary artifact hash."""
        if not self.cache_dir:
            return None
        return self.cache_dir / "vocabulary" / vocab_hash

    # ------------------------------------------------------------------
    # Static attribute resolution
    # ------------------------------------------------------------------

    def _build_source_static_slice(self, source: Source) -> pl.DataFrame | None:
        id_col = source.config.id_col
        static_cols = self._source_static_columns(source)
        if not static_cols:
            return None

        rename_map = {col: f"{source.name}__{col}" for col in static_cols}
        agg_exprs = [pl.col(col).first().alias(col) for col in static_cols]

        return (
            source.scan()
            .select([id_col, *static_cols])
            .filter(pl.col(id_col).is_in(self.entity_id_list))
            .group_by(id_col, maintain_order=True)
            .agg(agg_exprs)
            .select([id_col, *static_cols])
            .rename(rename_map)
            .rename({id_col: "entity_id"})
            .collect()
        )

    def _source_static_columns(self, source: Source) -> list[str]:
        cols: list[str] = []
        for group in (
            source.config.temporal_cols,
            source.config.categorical_cols,
            source.config.continuous_cols,
        ):
            if not group:
                continue
            cols.extend(col.col_name for col in group if col.static)
        return cols

    # ------------------------------------------------------------------
    # Split assignment helpers
    # ------------------------------------------------------------------

    def _assign_splits(self, entities: pl.DataFrame, cfg: CohortConfig) -> list[str]:
        if entities.height == 0:
            return []
        if not cfg.use_splits:
            return ["all"] * entities.height

        if cfg.stratify_col is None:
            return self._labels_for_indices(list(range(entities.height)), cfg, cfg.seed)

        if cfg.stratify_col not in entities.columns:
            raise ValueError(
                f"stratify_col '{cfg.stratify_col}' was not found in entities table columns: {entities.columns}"
            )

        stratum_values = entities.get_column(cfg.stratify_col).to_list()
        by_stratum: dict[str, list[int]] = {}
        for idx, value in enumerate(stratum_values):
            key = self._normalize_stratum_key(value)
            by_stratum.setdefault(key, []).append(idx)

        labels = [""] * entities.height
        for key, indices in by_stratum.items():
            stratum_seed = cfg.seed + int(self._stable_hash(key), 16)
            stratum_labels = self._labels_for_indices(indices, cfg, stratum_seed)
            for idx, split in zip(indices, stratum_labels):
                labels[idx] = split

        return labels

    def _labels_for_indices(
        self,
        indices: list[int],
        cfg: CohortConfig,
        seed: int,
    ) -> list[str]:
        n = len(indices)
        if n == 0:
            return []

        train_n = int(np.floor(n * cfg.train_frac))
        val_n = int(np.floor(n * cfg.val_frac))
        test_n = n - train_n - val_n

        labels = np.array(["test"] * n, dtype=object)
        labels[:train_n] = "train"
        labels[train_n : train_n + val_n] = "val"

        rng = np.random.default_rng(seed)
        rng.shuffle(labels)
        return labels.tolist()

    @staticmethod
    def _normalize_stratum_key(value: Any) -> str:
        return "<NULL>" if value is None else str(value)

    def _criteria_hash(self) -> str:
        payload = [criteria.model_dump(exclude_none=False) for criteria in self._criteria]
        return self._stable_hash(json.dumps(payload, sort_keys=True))

    @staticmethod
    def _stable_hash(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
