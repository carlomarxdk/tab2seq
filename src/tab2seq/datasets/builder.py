"""Build split-aware tokenized event datasets from a cohort and fitted tokenizer.

The ``EventDataset`` sits at the end of the pipeline::

    Vocabulary → Tokenizer → EventDataset

It delegates all tokenization strategy to :class:`~tab2seq.tokenization.Tokenizer`
and focuses purely on dataset construction: splitting, timestamp encoding,
relative date features, static embedding, persistence, and entity-level access.

Typical usage::

    vocab = Vocabulary(VocabularyConfig(max_vocab_size=50_000, min_token_count=5))
    vocab.fit_from_cohort_train(cohort)

    tok = Tokenizer(vocabulary=vocab)

    ds = EventDataset(cohort=cohort, tokenizer=tok, dataset_config=EventDatasetConfig(
        reference_date="1970-01-01",
        include_token_str=True,
    ))

    # Build + persist
    artifacts = ds.write_parquet()

    # Iterate train entities
    for record in ds.iter_entity_records("train", shuffle=True, seed=42):
        entity_id = record["entity_id"]
        events    = record["events"]    # list of dicts, each with token_ids
        static    = record["static"]    # dict of static attributes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Literal

RecordFormat = Literal["raw", "frame", "tensor", "padded_tensor"]

import numpy as np
import polars as pl

from tab2seq.cohort import Cohort, CohortConfig
from tab2seq.datasets.config import EventDatasetConfig
from tab2seq.datasets.registry import DatasetRegistry, DatasetRegistryEntry
from tab2seq.tokenization import Tokenizer, Vocabulary


@dataclass
class DatasetArtifacts:
    """Output paths for persisted event dataset artifacts.

    Attributes:
        dataset_dir: Root directory containing all dataset artifacts.
        metadata_path: Path to JSON metadata file (schema, row counts, config hashes).
        static_path: Path to Parquet file with static entity attributes.
        split_paths: Dict mapping split name → Parquet file path.
    """

    dataset_dir: Path
    metadata_path: Path
    static_path: Path
    split_paths: dict[str, Path]


class EventDataset:
    """Build split-aware tokenized event rows and persist to Parquet.

    Tokenization strategy is fully delegated to the injected
    :class:`~tab2seq.tokenization.Tokenizer`.  ``EventDataset`` handles:

    - Split-aware entity filtering (train / val / test)
    - Timestamp encoding (days since reference date)
    - Relative date features (e.g. age at event from birthdate)
    - Optional ``after_threshold`` flag column
    - Optional static attribute embedding into event rows
    - Parquet persistence with hash-based cache paths
    - In-memory split cache and O(1) entity index for fast retrieval

    Args:
        cohort: Cohort with source collection and optional cache directory.
        tokenizer: Fitted :class:`~tab2seq.tokenization.Tokenizer` instance.
            Split config is extracted from the tokenizer's vocabulary metadata
            to ensure consistency with vocabulary fitting.
        dataset_config: Dataset construction options.  Defaults to
            :class:`EventDatasetConfig`.
    """

    def __init__(
        self,
        cohort: Cohort,
        tokenizer: Tokenizer,
        dataset_config: EventDatasetConfig | None = None,
    ) -> None:
        self.cohort = cohort
        self.tokenizer = tokenizer
        self.split_config = self._split_config_from_vocab()

        self.dataset_config = dataset_config or EventDatasetConfig()

        self._require_fitted_tokenizer()

        # In-memory caches — keyed by split name
        self._split_cache: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
        self._split_entity_index: dict[
            str,
            tuple[dict[str, tuple[int, int]], dict[str, int], list[str]],
        ] = {}
        self._next_state: dict[tuple[str, bool, int | None], dict[str, Any]] = {}
        self._loaded_from_artifacts = False
        self._loaded_dataset_name: str | None = None
        self._loaded_dataset_dir: Path | None = None
        self._loaded_metadata: dict[str, Any] | None = None

    @classmethod
    def from_name(
        cls,
        name: str,
        registry_dir: str | Path,
    ) -> EventDataset:
        """Load a precomputed dataset by name from the dataset registry.

        This constructor loads split parquet files, static table, and vocabulary
        artifacts directly from disk. It does not scan sources, resolve cohort
        entities, assign splits, or rebuild vocabulary.

        Args:
            name: Registered dataset name.
            registry_dir: Directory containing ``registry.json``.

        Returns:
            Read-only ``EventDataset`` view backed by persisted artifacts.

        Raises:
            FileNotFoundError: If the dataset name or required artifacts are missing.
            ValueError: If registry or metadata contents are invalid.
        """
        registry = DatasetRegistry(Path(registry_dir))
        entry = registry.get(name)
        if entry is None:
            raise FileNotFoundError(
                f"Dataset name '{name}' not found in registry {registry.path}."
            )

        dataset_dir = Path(entry.dataset_dir)
        metadata_path = Path(entry.metadata_path)
        static_path = Path(entry.static_path)

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata is missing: {metadata_path}")
        if not static_path.exists():
            raise FileNotFoundError(f"Dataset static table is missing: {static_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_row_counts = metadata.get("split_row_counts")
        if not isinstance(split_row_counts, dict) or not split_row_counts:
            raise ValueError(
                f"Dataset metadata at {metadata_path} has no 'split_row_counts'."
            )

        split_paths: dict[str, Path] = {}
        for split_name in split_row_counts:
            split_path = dataset_dir / split_name / "part-000.parquet"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split parquet: {split_path}")
            split_paths[split_name] = split_path

        vocab_hash = entry.vocab_hash or metadata.get("vocab_hash")
        if vocab_hash is None:
            raise ValueError(
                f"Dataset metadata at {metadata_path} is missing 'vocab_hash'."
            )

        vocab_dir = (
            Path(entry.vocab_dir)
            if entry.vocab_dir is not None
            else cls._infer_vocab_dir(dataset_dir, vocab_hash)
        )
        vocabulary = cls._load_vocabulary(vocab_dir)
        tokenizer = Tokenizer(vocabulary=vocabulary)

        dataset = cls.__new__(cls)
        dataset.cohort = None
        dataset.tokenizer = tokenizer
        dataset.dataset_config = EventDatasetConfig(
            **metadata.get("dataset_config", {})
        )
        dataset.split_config = cls._load_split_config(dataset_dir, metadata, entry)

        dataset._split_cache = {}
        dataset._split_entity_index = {}
        dataset._next_state = {}
        dataset._loaded_from_artifacts = True
        dataset._loaded_dataset_name = name
        dataset._loaded_dataset_dir = dataset_dir
        dataset._loaded_metadata = metadata

        static_df = pl.read_parquet(static_path)
        if "split" not in static_df.columns:
            raise ValueError(
                f"Static table at {static_path} must include a 'split' column."
            )

        for split_name, split_path in split_paths.items():
            events_df = pl.read_parquet(split_path)
            split_static_df = static_df.filter(pl.col("split") == split_name)
            dataset._split_cache[split_name] = (events_df, split_static_df)
            dataset._split_entity_index[split_name] = dataset._build_entity_index(
                events_df,
                split_static_df,
            )

        return dataset

    # ------------------------------------------------------------------
    # Public build API
    # ------------------------------------------------------------------

    def build_split(
        self,
        split_name: str,
        force_recompute_splits: bool = False,
        split_df: pl.DataFrame | None = None,
        include_token_str: bool = True,
        embed_static_in_events: bool = False,
    ) -> pl.DataFrame:
        """Build tokenized event rows for one split.

        Each output row represents one event with columns:

        - ``entity_id``, ``split``, ``source_name``
        - ``primary_timestamp`` (ISO string), ``primary_time`` (days since reference)
        - ``token_ids`` (``List[Int64]``) — from :meth:`Tokenizer.encode_frame`
        - ``token_str`` (``Utf8``) — if ``include_token_str=True`` (default)
        - ``after_threshold`` (``Boolean``) — if ``dataset_config.include_after_threshold``
        - Relative date feature columns — per ``dataset_config.relative_date_features``

        Args:
            split_name: Split to build (``'train'``, ``'val'``, ``'test'``).
            force_recompute_splits: Clear caches and recompute splits from scratch.
            split_df: Pre-loaded split DataFrame to avoid redundant I/O.
            include_token_str: If ``True`` (default), add a human-readable ``token_str``
                column with space-joined token names.  Set to ``False`` to reduce
                dataset size when interpretability is not needed.
            embed_static_in_events: If ``True``, left-join the static entity table
                into every event row so static columns appear alongside event columns.
                Default ``False`` (static kept in a separate table).

        Returns:
            Polars DataFrame sorted by ``(entity_id, primary_timestamp, source_name)``.
        """
        if self._loaded_from_artifacts:
            if split_name in self._split_cache:
                return self._split_cache[split_name][0]
            raise ValueError(
                "Cannot build new splits from a dataset loaded with from_name(). "
                "Only persisted splits are available."
            )

        if force_recompute_splits:
            self._split_cache.clear()
            self._split_entity_index.clear()
            self._next_state.clear()

        if split_df is None:
            split_df = self.cohort.build_or_load_splits(
                self.split_config,
                force_recompute=force_recompute_splits,
            )

        available = set(split_df["split"].to_list())
        if split_name not in available:
            raise ValueError(
                f"Split '{split_name}' not found. Available: {sorted(available)}"
            )

        split_ids = set(
            split_df.filter(pl.col("split") == split_name)["entity_id"]
            .cast(pl.Utf8)
            .to_list()
        )

        static_df = self._build_static_table(split_df)
        split_static_df = static_df.filter(pl.col("split") == split_name)

        # O(1) static lookup for relative date feature computation
        static_lookup: dict[str, dict] = {
            str(d["entity_id"]): d for d in static_df.to_dicts()
        }

        ref_date = date.fromisoformat(self.dataset_config.reference_date)
        threshold_date = date.fromisoformat(self.dataset_config.threshold_date)

        source_frames: list[pl.DataFrame] = []

        for source in self.cohort.collection:
            primary_ts = source.config.primary_temporal
            if primary_ts is None:
                continue

            source_df = (
                source.scan()
                .filter(pl.col(source.config.id_col).cast(pl.Utf8).is_in(split_ids))
                .collect()
            )
            if source_df.is_empty():
                continue

            ts_col = primary_ts.col_name
            id_col = source.config.id_col
            src_name = source.name

            # Coerce timestamp once upfront
            source_df = source_df.with_columns(pl.col(ts_col).cast(pl.Date))

            # --- Tokenization (fully delegated to Tokenizer) ---
            source_df = self.tokenizer.encode_frame(
                source_df,
                source_name=src_name,
                columns=self._event_feature_columns(source),
                include_token_str=include_token_str,
            )

            # --- Scalar output columns ---
            source_df = source_df.with_columns(
                pl.col(id_col).cast(pl.Utf8).alias("entity_id"),
                pl.lit(split_name).alias("split"),
                pl.lit(src_name).alias("source_name"),
                pl.col(ts_col).cast(pl.Utf8).alias("primary_timestamp"),
                (pl.col(ts_col) - pl.lit(ref_date))
                .dt.total_days()
                .cast(pl.Int64)
                .alias("primary_time"),
            )

            if self.dataset_config.include_after_threshold:
                source_df = source_df.with_columns(
                    (pl.col(ts_col) >= pl.lit(threshold_date)).alias("after_threshold")
                )

            # --- Select final output columns ---
            out_cols = [
                "entity_id", "split", "source_name",
                "primary_timestamp", "primary_time", "token_ids",
            ]
            if include_token_str:
                out_cols.append("token_str")
            if self.dataset_config.include_after_threshold:
                out_cols.append("after_threshold")

            source_out = source_df.select(out_cols)

            # --- Relative date features (per-row dict lookup) ---
            if self.dataset_config.relative_date_features:
                rule_data: dict[str, list] = {
                    rule.output_column: []
                    for rule in self.dataset_config.relative_date_features
                }
                for row in source_out.iter_rows(named=True):
                    eid = row["entity_id"]
                    event_date = date.fromisoformat(row["primary_timestamp"])
                    entity_static = static_lookup.get(eid)
                    for rule in self.dataset_config.relative_date_features:
                        if entity_static is None:
                            rule_data[rule.output_column].append(None)
                            continue
                        raw = entity_static.get(rule.source_static_column)
                        if raw is None:
                            rule_data[rule.output_column].append(None)
                            continue
                        static_date = self._coerce_to_date(raw)
                        val = self._datetime_offset(event_date, static_date, rule.unit)
                        rule_data[rule.output_column].append(
                            int(val) if rule.floor_int else float(val)
                        )

                rule_cols = [
                    pl.Series(
                        rule.output_column,
                        rule_data[rule.output_column],
                        dtype=pl.Int64 if rule.floor_int else pl.Float64,
                    )
                    for rule in self.dataset_config.relative_date_features
                ]
                source_out = source_out.hstack(rule_cols)

            source_frames.append(source_out)

        events_df = (
            pl.concat(source_frames) if source_frames else self._empty_events_frame(include_token_str)
        )

        if embed_static_in_events:
            events_df = events_df.join(split_static_df, on="entity_id", how="left")

        events_df = events_df.sort(["entity_id", "primary_timestamp", "source_name"])

        self._split_cache[split_name] = (events_df, split_static_df)
        self._split_entity_index[split_name] = self._build_entity_index(
            events_df, split_static_df
        )
        return events_df

    def build_all_splits(
        self,
        force_recompute_splits: bool = False,
        include_token_str: bool = True,
        embed_static_in_events: bool = False,
    ) -> dict[str, pl.DataFrame]:
        """Build tokenized event rows for all available splits.

        Args:
            force_recompute_splits: Clear caches and recompute from scratch.
            include_token_str: See :meth:`build_split`.
            embed_static_in_events: See :meth:`build_split`.

        Returns:
            Dict mapping split name → tokenized event DataFrame.
        """
        if self._loaded_from_artifacts:
            return {name: bundle[0] for name, bundle in self._split_cache.items()}

        split_df = self.cohort.build_or_load_splits(
            self.split_config,
            force_recompute=force_recompute_splits,
        )
        split_names = sorted(set(split_df["split"].to_list()))
        return {
            name: self.build_split(
                name,
                force_recompute_splits=force_recompute_splits,
                split_df=split_df,
                include_token_str=include_token_str,
                embed_static_in_events=embed_static_in_events,
            )
            for name in split_names
        }

    def build_static_table(self, force_recompute_splits: bool = False) -> pl.DataFrame:
        """Build the static entity table (entity_id + split + static columns).

        Args:
            force_recompute_splits: Clear caches and recompute from scratch.

        Returns:
            One row per entity with static attributes and split assignment.
        """
        if self._loaded_from_artifacts:
            if not self._split_cache:
                return pl.DataFrame(
                    {
                        "entity_id": pl.Series([], dtype=pl.Utf8),
                        "split": pl.Series([], dtype=pl.Utf8),
                    }
                )
            frames = [bundle[1] for bundle in self._split_cache.values()]
            return pl.concat(frames).sort("entity_id")

        split_df = self.cohort.build_or_load_splits(
            self.split_config,
            force_recompute=force_recompute_splits,
        )
        return self._build_static_table(split_df)

    def write_parquet(
        self,
        force_recompute_splits: bool = False,
        dataset_name: str | None = None,
        registry_dir: str | Path | None = None,
        overwrite_name: bool = False,
        force_write: bool | None = None,
        include_token_str: bool = True,
        embed_static_in_events: bool = False,
    ) -> DatasetArtifacts:
        """Persist all split event datasets and the static table as Parquet.

        Output layout::

            {dataset_root}/
              metadata.json
              static/entities_static.parquet
              train/part-000.parquet
              val/part-000.parquet
              test/part-000.parquet

        The root path is determined by ``dataset_config.output_dir`` if set,
        otherwise by ``cohort.cache_dir``.

        Args:
            force_recompute_splits: Clear caches and recompute from scratch.
            dataset_name: Optional human-readable name to register after write.
            registry_dir: Optional registry directory. Defaults to cohort datasets dir.
            overwrite_name: Backward-compatible alias for force_write.
            force_write: If True, replace an existing dataset name registration.
                If False, raise if the name already exists.
            include_token_str: See :meth:`build_split`.
            embed_static_in_events: See :meth:`build_split`.

        Returns:
            :class:`DatasetArtifacts` with all output paths.
        """
        if self._loaded_from_artifacts:
            raise ValueError(
                "Cannot write_parquet() from a dataset loaded via from_name()."
            )

        splits = self.build_all_splits(
            force_recompute_splits=force_recompute_splits,
            include_token_str=include_token_str,
            embed_static_in_events=embed_static_in_events,
        )
        static_df = self.build_static_table(force_recompute_splits=force_recompute_splits)

        dataset_hash = self._dataset_hash()
        root = self._dataset_root(dataset_hash)
        root.mkdir(parents=True, exist_ok=True)

        split_paths: dict[str, Path] = {}
        for split_name, frame in splits.items():
            split_dir = root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            path = split_dir / "part-000.parquet"
            frame.write_parquet(path)
            split_paths[split_name] = path

        static_dir = root / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        static_path = static_dir / "entities_static.parquet"
        static_df.write_parquet(static_path)

        metadata = {
            "dataset_hash": dataset_hash,
            "cohort_name": self.cohort.name,
            "split_hash": self.split_config.config_hash(),
            "split_config": self.split_config.model_dump(exclude_none=False),
            "vocab_hash": (self.tokenizer.vocabulary.metadata or {}).get("vocab_hash"),
            "dataset_config": self.dataset_config.model_dump(mode="json"),
            "event_schema": (
                {k: str(v) for k, v in splits[next(iter(splits))].schema.items()}
                if splits else {}
            ),
            "static_schema": {k: str(v) for k, v in static_df.schema.items()},
            "split_row_counts": {name: frame.height for name, frame in splits.items()},
        }
        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        artifacts = DatasetArtifacts(
            dataset_dir=root,
            metadata_path=metadata_path,
            static_path=static_path,
            split_paths=split_paths,
        )

        if dataset_name is not None:
            should_overwrite = force_write if force_write is not None else overwrite_name
            self.register_name(
                dataset_name,
                registry_dir=registry_dir,
                overwrite=should_overwrite,
                artifacts=artifacts,
            )

        return artifacts

    def register_name(
        self,
        name: str,
        registry_dir: str | Path | None = None,
        overwrite: bool = False,
        artifacts: DatasetArtifacts | None = None,
        force_write: bool | None = None,
    ) -> DatasetRegistryEntry:
        """Register this dataset under a human-readable name.

        Args:
            name: Name to register.
            registry_dir: Registry directory containing ``registry.json``.
            overwrite: Backward-compatible alias for force_write.
            artifacts: Optional artifacts object from ``write_parquet``.
            force_write: If True, replace an existing dataset name registration.
                If False, raise if the name already exists.

        Returns:
            The registry entry that was saved.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        registry_base = self._resolve_registry_dir(registry_dir)
        registry = DatasetRegistry(registry_base)

        if artifacts is None:
            if self._loaded_dataset_dir is not None:
                dataset_dir = self._loaded_dataset_dir
            elif self._loaded_metadata is not None:
                dataset_hash = str(self._loaded_metadata.get("dataset_hash"))
                dataset_dir = self._dataset_root(dataset_hash)
            else:
                dataset_hash = self._dataset_hash()
                dataset_dir = self._dataset_root(dataset_hash)

            artifacts = self._artifacts_from_dataset_dir(dataset_dir)

        metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
        split_hash = metadata.get("split_hash")
        vocab_hash = metadata.get("vocab_hash")
        cohort_name = metadata.get("cohort_name")

        vocab_dir: Path | None = None
        if vocab_hash is not None:
            if self.cohort is not None and self.cohort.cache_dir is not None:
                vocab_dir = self.cohort.vocabulary_cache_dir(vocab_hash)
            elif self._loaded_dataset_dir is not None:
                vocab_dir = self._infer_vocab_dir(self._loaded_dataset_dir, str(vocab_hash))

        entry = DatasetRegistryEntry(
            name=name.strip(),
            dataset_hash=str(metadata.get("dataset_hash")),
            dataset_dir=str(artifacts.dataset_dir),
            metadata_path=str(artifacts.metadata_path),
            static_path=str(artifacts.static_path),
            split_hash=str(split_hash) if split_hash is not None else None,
            vocab_hash=str(vocab_hash) if vocab_hash is not None else None,
            vocab_dir=str(vocab_dir) if vocab_dir is not None else None,
            cohort_name=str(cohort_name) if cohort_name is not None else None,
            created_at_utc=DatasetRegistry.now_utc_iso(),
        )
        should_overwrite = force_write if force_write is not None else overwrite
        registry.register(entry, overwrite=should_overwrite)
        return entry

    # ------------------------------------------------------------------
    # Entity-level access API
    # ------------------------------------------------------------------

    def get_entity_record(
        self,
        entity_id: str,
        split: str,
        format: RecordFormat = "raw",
        pad_id: int = 0,
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
        include_after_threshold: bool = False,
        censoring: Literal["none", "left", "right"] = "none",
        max_events: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Return one entity payload with static attributes and event rows.

        Args:
            entity_id: Entity ID to retrieve.
            split: Split name (``'train'``, ``'val'``, ``'test'``).
            format: Output format. One of:

                - ``'raw'`` *(default)* — nested Python dicts, one dict per event.
                - ``'frame'`` — raw Polars DataFrames; avoids ``to_dicts()`` overhead.
                - ``'tensor'`` — NumPy arrays with flat concatenated token IDs,
                  per-event lengths, ``time`` (int64 primary timestamps),
                  a ``[num_events, T]`` temporal matrix, and static token IDs.
                - ``'padded_tensor'`` — like ``'tensor'`` but token IDs are a
                  ``[num_events, max_event_len]`` matrix padded with ``pad_id``,
                  plus a matching boolean ``attention_mask``.

            pad_id: Padding value for ``'padded_tensor'`` format (default ``0``).
            include_cls: If ``True`` (default), prepend ``[CLS]`` to the static
                token sequence.  When ``static_as_event=False`` (default): prepended
                to ``static_token_ids`` / ``static["token_ids"]`` (and the
                corresponding ``token_str`` fields for ``'raw'``/``'frame'``).
                When ``static_as_event=True``: ``[CLS]`` becomes the first token
                of event 0 (the static event).
            include_sep: If ``True`` (default), append ``[SEP]`` to each event's
                token IDs.  When ``static_as_event=True``, ``[SEP]`` is also
                appended to the static event (event 0).
            static_as_event: If ``True``, embed static tokens as event 0 instead
                of returning them in a separate ``static_token_ids`` field.
                Event 0 has ``primary_time=0`` and zero-valued temporal features.
                ``[CLS]`` / ``[SEP]`` (when enabled) are included in the token
                sequence of event 0.  ``static_token_ids`` is always populated
                regardless of this flag.  Default: ``False``.
            include_after_threshold: If ``False`` (default), exclude events with
                ``after_threshold=True`` from the returned sequence.  Has no effect
                if the ``after_threshold`` column was not added at build time
                (``EventDatasetConfig.include_after_threshold=False``).
            censoring: How to truncate when a limit is exceeded.
                ``'none'`` (default) — no truncation.
                ``'right'`` — keep the earliest events (drop from the end).
                ``'left'`` — keep the latest events (drop from the beginning).
                Requires ``max_events`` or ``max_tokens`` to have any effect.
            max_events: Maximum number of events to return per entity.  When the
                entity has more events, ``censoring`` determines which to drop.
                Cannot be combined with ``max_tokens``.
            max_tokens: Maximum total token count across all events.  Full events
                are removed (never partial) until the total fits within the limit.
                ``censoring`` determines which events to drop.
                Cannot be combined with ``max_events``.

        Returns:
            Dict with format-dependent keys, or ``None`` if the entity is not found.

            ``'raw'``: ``entity_id``, ``split``, ``static`` (dict), ``events`` (list of dicts).
            When ``static_as_event=True``, ``events[0]`` is the static event with
            ``source_name='__static__'`` and ``primary_time=0``.

            ``'frame'``: ``entity_id``, ``split``, ``events`` (DataFrame with
            ``token_lengths`` column added), ``static`` (DataFrame),
            ``static_token_ids`` (list), ``static_token_str`` (str).

            ``'tensor'``: ``entity_id``, ``split``, ``token_ids`` (int64 [total_tokens]),
            ``token_str`` (list[str] or None), ``event_lengths`` (int64 [E]),
            ``time`` (int64 [E] — primary timestamps; same values as ``temporal[:,0]``
            but kept as integers), ``temporal`` (float64 [E, T]),
            ``static_token_ids`` (int64 [S]), ``static_token_str`` (str).

            ``'padded_tensor'``: same as ``'tensor'`` but ``token_ids`` is
            int64 [E, max_len] and ``attention_mask`` (bool [E, max_len]) is
            added; ``event_lengths`` is omitted.
        """
        events_df, static_df = self._get_split_bundle(split)
        event_ranges, static_index, _ = self._split_entity_index[split]
        entity_id = str(entity_id)

        event_slice = event_ranges.get(entity_id)
        events_slice = (
            events_df.slice(*event_slice)
            if event_slice is not None
            else events_df.clear()
        )

        static_row_idx = static_index.get(entity_id)
        static_slice = (
            static_df.slice(static_row_idx, 1)
            if static_row_idx is not None
            else static_df.clear()
        )

        if events_slice.is_empty() and static_slice.is_empty():
            return None

        if not include_after_threshold and "after_threshold" in events_slice.columns:
            events_slice = events_slice.filter(~pl.col("after_threshold"))

        events_slice = self._apply_event_censoring(events_slice, censoring, max_events, max_tokens)

        static_payload = self._compute_static_tokens(static_slice, entity_id, split)

        if format == "raw":
            return self._as_raw(entity_id, split, events_slice, static_payload, include_cls, include_sep, static_as_event)
        if format == "frame":
            return self._as_frame(entity_id, split, events_slice, static_slice, static_payload, include_cls, include_sep, static_as_event)
        if format == "tensor":
            return self._as_tensor(entity_id, split, events_slice, static_payload, include_cls, include_sep, static_as_event)
        if format == "padded_tensor":
            return self._as_padded_tensor(entity_id, split, events_slice, static_payload, pad_id, include_cls, include_sep, static_as_event)
        raise ValueError(
            f"Unknown format {format!r}. Expected 'raw', 'frame', 'tensor', or 'padded_tensor'."
        )

    def sample_entity_record(
        self,
        split: str,
        seed: int | None = None,
        format: RecordFormat = "raw",
        pad_id: int = 0,
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
        include_after_threshold: bool = False,
        censoring: Literal["none", "left", "right"] = "none",
        max_events: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Return one randomly sampled entity record from a split.

        Args:
            split: Split name.
            seed: Optional random seed for reproducibility.
            format: Output format — see :meth:`get_entity_record` for options.
            pad_id: Padding value used when ``format='padded_tensor'``.
            include_cls: Prepend ``[CLS]`` — see :meth:`get_entity_record`.
            include_sep: Append ``[SEP]`` to each event — see :meth:`get_entity_record`.
            static_as_event: Embed static as event 0 — see :meth:`get_entity_record`.
            include_after_threshold: Include post-threshold events — see :meth:`get_entity_record`.
            censoring: Truncation strategy — see :meth:`get_entity_record`.
            max_events: Event count limit — see :meth:`get_entity_record`.
            max_tokens: Token count limit — see :meth:`get_entity_record`.

        Returns:
            Entity record dict, or ``None`` if split is empty.
        """
        self._get_split_bundle(split)
        entity_ids = self._entity_order(split)
        if not entity_ids:
            return None
        rng = np.random.default_rng(seed)
        chosen = str(rng.choice(np.array(entity_ids, dtype=object)))
        return self.get_entity_record(
            chosen, split=split, format=format, pad_id=pad_id,
            include_cls=include_cls, include_sep=include_sep, static_as_event=static_as_event,
            include_after_threshold=include_after_threshold,
            censoring=censoring, max_events=max_events, max_tokens=max_tokens,
        )

    def iter_entity_records(
        self,
        split: str,
        shuffle: bool = False,
        seed: int | None = None,
        format: RecordFormat = "raw",
        pad_id: int = 0,
        include_cls: bool = True,
        include_sep: bool = True,
        include_after_threshold: bool = False,
        censoring: Literal["none", "left", "right"] = "none",
        max_events: int | None = None,
        max_tokens: int | None = None,
        static_as_event: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Iterate all entity records in a split.

        Args:
            split: Split name.
            shuffle: Shuffle entity order before iteration.
            seed: Optional random seed for reproducibility.
            format: Output format — see :meth:`get_entity_record` for options.
            pad_id: Padding value used when ``format='padded_tensor'``.
            include_cls: Prepend ``[CLS]`` — see :meth:`get_entity_record`.
            include_sep: Append ``[SEP]`` to each event — see :meth:`get_entity_record`.
            static_as_event: Embed static as event 0 — see :meth:`get_entity_record`.
            include_after_threshold: Include post-threshold events — see :meth:`get_entity_record`.
            censoring: Truncation strategy — see :meth:`get_entity_record`.
            max_events: Event count limit — see :meth:`get_entity_record`.
            max_tokens: Token count limit — see :meth:`get_entity_record`.

        Yields:
            Entity record dicts.
        """
        self._get_split_bundle(split)
        entity_ids = self._entity_order(split)

        if shuffle and entity_ids:
            rng = np.random.default_rng(seed)
            entity_ids = [
                str(v)
                for v in rng.permutation(np.array(entity_ids, dtype=object)).tolist()
            ]

        for entity_id in entity_ids:
            record = self.get_entity_record(
                str(entity_id), split=split, format=format, pad_id=pad_id,
                include_cls=include_cls, include_sep=include_sep, static_as_event=static_as_event,
                include_after_threshold=include_after_threshold,
                censoring=censoring, max_events=max_events, max_tokens=max_tokens,
            )
            if record is not None:
                yield record

    def next_entity_record(
        self,
        split: str,
        shuffle: bool = False,
        seed: int | None = None,
        reset: bool = False,
        format: RecordFormat = "raw",
        pad_id: int = 0,
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
        include_after_threshold: bool = False,
        censoring: Literal["none", "left", "right"] = "none",
        max_events: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the next entity record in sweep order, or ``None`` when exhausted.

        State is maintained per ``(split, shuffle, seed)`` key.  Call with
        ``reset=True`` to restart iteration from the beginning.

        Args:
            split: Split name.
            shuffle: Shuffle entity order.
            seed: Optional random seed for reproducibility.
            reset: Reset iteration state for this split/shuffle/seed combination.
            format: Output format — see :meth:`get_entity_record` for options.
            pad_id: Padding value used when ``format='padded_tensor'``.
            include_cls: Prepend ``[CLS]`` — see :meth:`get_entity_record`.
            include_sep: Append ``[SEP]`` to each event — see :meth:`get_entity_record`.
            static_as_event: Embed static as event 0 — see :meth:`get_entity_record`.
            include_after_threshold: Include post-threshold events — see :meth:`get_entity_record`.
            censoring: Truncation strategy — see :meth:`get_entity_record`.
            max_events: Event count limit — see :meth:`get_entity_record`.
            max_tokens: Token count limit — see :meth:`get_entity_record`.

        Returns:
            Entity record dict, or ``None`` when all entities have been returned.
        """
        key = (split, shuffle, seed)

        if reset or key not in self._next_state:
            entity_ids = self._entity_order(split)
            if shuffle and entity_ids:
                rng = np.random.default_rng(seed)
                entity_ids = [
                    str(v)
                    for v in rng.permutation(np.array(entity_ids, dtype=object)).tolist()
                ]
            self._next_state[key] = {"order": entity_ids, "index": 0}

        state = self._next_state[key]
        index = int(state["index"])
        if index >= len(state["order"]):
            return None

        entity_id = str(state["order"][index])
        state["index"] = index + 1
        return self.get_entity_record(
            entity_id, split=split, format=format, pad_id=pad_id,
            include_cls=include_cls, include_sep=include_sep, static_as_event=static_as_event,
            include_after_threshold=include_after_threshold,
            censoring=censoring, max_events=max_events, max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Private — record formatting
    # ------------------------------------------------------------------

    def _apply_event_censoring(
        self,
        events_slice: pl.DataFrame,
        censoring: Literal["none", "left", "right"],
        max_events: int | None,
        max_tokens: int | None,
    ) -> pl.DataFrame:
        """Truncate events_slice according to censoring strategy and limits.

        ``censoring='none'`` always returns the slice unchanged.
        ``censoring='right'`` keeps the earliest events (drop tail).
        ``censoring='left'`` keeps the latest events (drop head).
        Exactly one of ``max_events`` / ``max_tokens`` may be set; passing both
        raises ``ValueError``.  Token counts are per-event list lengths; whole
        events are always dropped — never partial.
        """
        if max_events is not None and max_tokens is not None:
            raise ValueError("Specify either max_events or max_tokens, not both.")
        if censoring == "none" or (max_events is None and max_tokens is None):
            return events_slice

        if max_events is not None:
            n = len(events_slice)
            if n <= max_events:
                return events_slice
            return events_slice.tail(max_events) if censoring == "left" else events_slice.head(max_events)

        # max_tokens path — drop full events until total fits
        token_limit: int = max_tokens  # type: ignore[assignment]  # narrowed above
        lengths: list[int] = events_slice["token_ids"].list.len().to_list()
        if censoring == "right":
            cumsum, n_keep = 0, 0
            for length in lengths:
                if cumsum + length > token_limit:
                    break
                cumsum += length
                n_keep += 1
            return events_slice.head(n_keep)
        else:  # censoring == "left"
            cumsum, n_keep = 0, 0
            for length in reversed(lengths):
                if cumsum + length > token_limit:
                    break
                cumsum += length
                n_keep += 1
            return events_slice.tail(n_keep)

    def _compute_static_tokens(
        self,
        static_slice: pl.DataFrame,
        entity_id: str,
        split: str,
    ) -> dict[str, Any]:
        """Build static entity payload with tokenized attributes."""
        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "split": split,
            "token_ids": [],
            "token_str": "",
        }
        if static_slice.is_empty():
            return payload

        row = static_slice.to_dicts()[0]
        payload.update(row)

        feature_keys = sorted(
            key
            for key in payload
            if key not in {"entity_id", "split", "token_ids", "token_str"}
        )
        unk_id = self.tokenizer.token2index.get(self.tokenizer.vocabulary.config.unk_token)
        if unk_id is None:
            raise ValueError("Tokenizer vocabulary is missing the UNK special token.")

        tokens: list[str] = []
        token_ids: list[int] = []
        prefix_cache: dict[str, dict[str, str]] = {}
        for key in feature_keys:
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, (date, datetime, np.datetime64)):
                continue
            if isinstance(value, str):
                try:
                    date.fromisoformat(value)
                    continue
                except ValueError:
                    pass
                try:
                    datetime.fromisoformat(value)
                    continue
                except ValueError:
                    pass
            source_name, sep, col_name = key.partition("__")
            if sep:
                if source_name not in prefix_cache:
                    prefix_cache[source_name] = self.tokenizer.vocabulary.column_prefixes(source_name)
                prefix = prefix_cache[source_name].get(col_name, col_name)
                token = f"{source_name}__{prefix}__{value}"
            else:
                token = f"{key}__{value}"
            tokens.append(token)
            token_ids.append(self.tokenizer.token2index.get(token, unk_id))

        payload["token_ids"] = token_ids
        payload["token_str"] = " ".join(tokens)
        return payload

    def _as_raw(
        self,
        entity_id: str,
        split: str,
        events_slice: pl.DataFrame,
        static_payload: dict[str, Any],
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
        
    ) -> dict[str, Any]:
        """Return entity record as nested Python dicts (default behavior)."""
        sep_str = self.tokenizer.vocabulary.config.sep_token
        cls_str = self.tokenizer.vocabulary.config.cls_token
        events = events_slice.to_dicts()
        if include_sep:
            sep_id = self._sep_id()
            for event in events:
                event["token_ids"] = list(event["token_ids"]) + [sep_id]
                if "token_str" in event:
                    ts = event["token_str"]
                    event["token_str"] = (ts + " " + sep_str) if ts else sep_str
        if static_as_event:
            static_ids: list[int] = list(static_payload["token_ids"])
            if include_cls:
                static_ids = [self._cls_id()] + static_ids
            if include_sep:
                static_ids = static_ids + [self._sep_id()]
            static_event: dict[str, Any] = {
                "entity_id": entity_id,
                "split": split,
                "source_name": "__static__",
                "primary_timestamp": None,
                "primary_time": 0,
                "token_ids": static_ids,
            }
            if "token_str" in events_slice.columns:
                ts = static_payload.get("token_str", "")
                if include_cls:
                    ts = (cls_str + " " + ts) if ts else cls_str
                if include_sep:
                    ts = (ts + " " + sep_str) if ts else sep_str
                static_event["token_str"] = ts
            events = [static_event] + events
            # Apply CLS to static_out token fields (same as static_as_event=False path)
            static_out = dict(static_payload)
            if include_cls:
                static_out["token_ids"] = [self._cls_id()] + list(static_out["token_ids"])
                ts = static_out.get("token_str", "")
                static_out["token_str"] = (cls_str + " " + ts) if ts else cls_str
        else:
            if include_cls:
                static_payload = dict(static_payload)
                static_payload["token_ids"] = [self._cls_id()] + list(static_payload["token_ids"])
                ts = static_payload.get("token_str", "")
                static_payload["token_str"] = (cls_str + " " + ts) if ts else cls_str
            static_out = static_payload
        return {
            "entity_id": entity_id,
            "split": split,
            "static": static_out,
            "events": events,
        }

    def _as_frame(
        self,
        entity_id: str,
        split: str,
        events_slice: pl.DataFrame,
        static_slice: pl.DataFrame,
        static_payload: dict[str, Any],
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
    ) -> dict[str, Any]:
        """Return entity record with raw Polars DataFrames; avoids to_dicts() overhead."""
        sep_str = self.tokenizer.vocabulary.config.sep_token
        cls_str = self.tokenizer.vocabulary.config.cls_token
        if include_sep:
            sep_id = self._sep_id()
            n = len(events_slice)
            events_slice = events_slice.with_columns(
                pl.col("token_ids").list.concat(
                    pl.Series("", [[sep_id]] * n, dtype=pl.List(pl.Int64))
                )
            )
            if "token_str" in events_slice.columns:
                events_slice = events_slice.with_columns(
                    pl.when(pl.col("token_str").str.len_chars() > 0)
                    .then(pl.col("token_str") + f" {sep_str}")
                    .otherwise(pl.lit(sep_str))
                    .alias("token_str")
                )
        if static_as_event:
            static_ids: list[int] = list(static_payload["token_ids"])
            if include_cls:
                static_ids = [self._cls_id()] + static_ids
            if include_sep:
                static_ids = static_ids + [self._sep_id()]
            schema = events_slice.schema
            row: dict[str, Any] = {col: [None] for col in schema}
            row["entity_id"] = [entity_id]
            row["split"] = [split]
            row["source_name"] = ["__static__"]
            row["primary_timestamp"] = [None]
            row["primary_time"] = [0]
            row["token_ids"] = [static_ids]
            if "token_str" in schema:
                ts = static_payload.get("token_str", "")
                if include_cls:
                    ts = (cls_str + " " + ts) if ts else cls_str
                if include_sep:
                    ts = (ts + " " + sep_str) if ts else sep_str
                row["token_str"] = [ts]
            static_event_df = pl.DataFrame(
                {col: pl.Series([row[col][0]], dtype=schema[col]) for col in schema}
            )
            events_slice = pl.concat([static_event_df, events_slice])
            static_token_ids: list[int] = list(static_payload["token_ids"])
            static_token_str = static_payload["token_str"]
            if include_cls:
                static_token_ids = [self._cls_id()] + static_token_ids
                static_token_str = (cls_str + " " + static_token_str) if static_token_str else cls_str
        else:
            static_token_ids = list(static_payload["token_ids"])
            static_token_str = static_payload["token_str"]
            if include_cls:
                static_token_ids = [self._cls_id()] + static_token_ids
                static_token_str = (cls_str + " " + static_token_str) if static_token_str else cls_str
        events_slice = events_slice.with_columns(
            pl.col("token_ids").list.len().cast(pl.Int64).alias("token_lengths")
        )
        return {
            "entity_id": entity_id,
            "split": split,
            "events": events_slice,
            "static": static_slice,
            "static_token_ids": static_token_ids,
            "static_token_str": static_token_str,
        }

    def _as_tensor(
        self,
        entity_id: str,
        split: str,
        events_slice: pl.DataFrame,
        static_payload: dict[str, Any],
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
    ) -> dict[str, Any]:
        """Return entity record as flat NumPy arrays suitable for model input."""
        token_id_lists: list[list[int]] = events_slice["token_ids"].to_list()

        if include_sep:
            sep_id = self._sep_id()
            token_id_lists = [t + [sep_id] for t in token_id_lists]

        time = events_slice["primary_time"].to_numpy().astype(np.int64)
        temporal = self._build_temporal_array(events_slice)

        raw_static = np.array(static_payload["token_ids"], dtype=np.int64)
        static_token_ids = (
            np.concatenate([[self._cls_id()], raw_static]) if include_cls else raw_static
        )

        if static_as_event:
            static_ids: list[int] = list(static_payload["token_ids"])
            if include_cls:
                static_ids = [self._cls_id()] + static_ids
            if include_sep:
                static_ids = static_ids + [self._sep_id()]
            all_lists = [static_ids] + token_id_lists
            time = np.concatenate([[0], time])
            temporal = np.vstack([np.zeros((1, temporal.shape[1]), dtype=np.float64), temporal])
        else:
            all_lists = token_id_lists

        event_lengths = np.array([len(t) for t in all_lists], dtype=np.int64)
        flat_tokens = (
            np.concatenate(all_lists).astype(np.int64)
            if all_lists
            else np.array([], dtype=np.int64)
        )

        sep_str = self.tokenizer.vocabulary.config.sep_token
        cls_str = self.tokenizer.vocabulary.config.cls_token
        if "token_str" in events_slice.columns:
            strs: list[str] = events_slice["token_str"].to_list()
            if include_sep:
                strs = [(s + " " + sep_str) if s else sep_str for s in strs]
            if static_as_event:
                raw_s = static_payload.get("token_str", "")
                if include_cls:
                    raw_s = (cls_str + " " + raw_s) if raw_s else cls_str
                if include_sep:
                    raw_s = (raw_s + " " + sep_str) if raw_s else sep_str
                strs = [raw_s] + strs
            token_str_out: list[str] | None = strs
        else:
            token_str_out = None

        raw_static_str = static_payload.get("token_str", "")
        static_token_str = (
            ((cls_str + " " + raw_static_str) if raw_static_str else cls_str)
            if include_cls else raw_static_str
        )

        return {
            "entity_id": entity_id,
            "split": split,
            "token_ids": flat_tokens,
            "token_str": token_str_out,
            "event_lengths": event_lengths,
            "time": time,
            "temporal": temporal,
            "static_token_ids": static_token_ids,
            "static_token_str": static_token_str,
        }

    def _as_padded_tensor(
        self,
        entity_id: str,
        split: str,
        events_slice: pl.DataFrame,
        static_payload: dict[str, Any],
        pad_id: int = 0,
        include_cls: bool = True,
        include_sep: bool = True,
        static_as_event: bool = False,
    ) -> dict[str, Any]:
        """Return entity record as padded 2-D NumPy arrays for batched model input."""
        token_id_lists: list[list[int]] = events_slice["token_ids"].to_list()

        if include_sep:
            sep_id = self._sep_id()
            token_id_lists = [t + [sep_id] for t in token_id_lists]

        time = events_slice["primary_time"].to_numpy().astype(np.int64)
        temporal = self._build_temporal_array(events_slice)

        raw_static = np.array(static_payload["token_ids"], dtype=np.int64)
        static_token_ids = (
            np.concatenate([[self._cls_id()], raw_static]) if include_cls else raw_static
        )

        if static_as_event:
            static_ids = list(static_payload["token_ids"])
            if include_cls:
                static_ids = [self._cls_id()] + static_ids
            if include_sep:
                static_ids = static_ids + [self._sep_id()]
            all_lists = [static_ids] + token_id_lists
            time = np.concatenate([[0], time])
            temporal = np.vstack([np.zeros((1, temporal.shape[1]), dtype=np.float64), temporal])
        else:
            all_lists = token_id_lists

        num_rows = len(all_lists)
        max_len = max((len(t) for t in all_lists), default=0)

        token_matrix = np.full((num_rows, max_len), fill_value=pad_id, dtype=np.int64)
        attention_mask = np.zeros((num_rows, max_len), dtype=bool)
        for i, tokens in enumerate(all_lists):
            n = len(tokens)
            token_matrix[i, :n] = tokens
            attention_mask[i, :n] = True

        sep_str = self.tokenizer.vocabulary.config.sep_token
        cls_str = self.tokenizer.vocabulary.config.cls_token
        if "token_str" in events_slice.columns:
            strs: list[str] = events_slice["token_str"].to_list()
            if include_sep:
                strs = [(s + " " + sep_str) if s else sep_str for s in strs]
            if static_as_event:
                raw_s = static_payload.get("token_str", "")
                if include_cls:
                    raw_s = (cls_str + " " + raw_s) if raw_s else cls_str
                if include_sep:
                    raw_s = (raw_s + " " + sep_str) if raw_s else sep_str
                strs = [raw_s] + strs
            token_str_out: list[str] | None = strs
        else:
            token_str_out = None

        raw_static_str = static_payload.get("token_str", "")
        static_token_str = (
            ((cls_str + " " + raw_static_str) if raw_static_str else cls_str)
            if include_cls else raw_static_str
        )

        return {
            "entity_id": entity_id,
            "split": split,
            "token_ids": token_matrix,
            "token_str": token_str_out,
            "attention_mask": attention_mask,
            "time": time,
            "temporal": temporal,
            "static_token_ids": static_token_ids,
            "static_token_str": static_token_str,
        }

    def _cls_id(self) -> int:
        token = self.tokenizer.vocabulary.config.cls_token
        t2i = self.tokenizer.token2index
        if token not in t2i:
            raise KeyError(f"Special token '{token}' missing from vocabulary.")
        return t2i[token]

    def _sep_id(self) -> int:
        token = self.tokenizer.vocabulary.config.sep_token
        t2i = self.tokenizer.token2index
        if token not in t2i:
            raise KeyError(f"Special token '{token}' missing from vocabulary.")
        return t2i[token]

    def _build_temporal_array(self, events_slice: pl.DataFrame) -> np.ndarray:
        """Build float64 [num_events, T] temporal feature matrix.

        Columns: ``primary_time`` followed by each relative-date-feature output column
        in the order they appear in ``dataset_config.relative_date_features``.
        """
        n = len(events_slice)
        rel_cols = [rule.output_column for rule in self.dataset_config.relative_date_features]
        if n == 0:
            return np.zeros((0, 1 + len(rel_cols)), dtype=np.float64)
        parts = [events_slice["primary_time"].to_numpy()]
        for col in rel_cols:
            parts.append(events_slice[col].to_numpy())
        return np.column_stack(parts).astype(np.float64)

    # ------------------------------------------------------------------
    # Private — caching and indexing
    # ------------------------------------------------------------------

    def _split_config_from_vocab(self) -> CohortConfig:
        """Load the split config that was used when fitting the vocabulary.

        This ensures EventDataset always uses the same train/val/test assignment
        as the vocabulary — preventing silent leakage from split mismatch.
        """
        meta = self.tokenizer.vocabulary.metadata
        if meta is None:
            raise ValueError("Vocabulary metadata missing — fit vocabulary first.")
        split_hash = meta.get("split_hash")
        if split_hash is None:
            raise ValueError("Vocabulary metadata missing 'split_hash'.")
        return self.cohort.load_split_config(split_hash)
        
    def _get_split_bundle(
        self,
        split: str,
        force_recompute_splits: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if force_recompute_splits:
            self._split_cache.pop(split, None)
            self._split_entity_index.pop(split, None)
        if split not in self._split_cache:
            self.build_split(split, force_recompute_splits=force_recompute_splits)
        if split not in self._split_entity_index:
            events_df, static_df = self._split_cache[split]
            self._split_entity_index[split] = self._build_entity_index(
                events_df, static_df
            )
        return self._split_cache[split]

    def _entity_order(self, split: str) -> list[str]:
        if split not in self._split_entity_index:
            self._get_split_bundle(split)
        return self._split_entity_index[split][2]

    def _build_entity_index(
        self,
        events_df: pl.DataFrame,
        static_df: pl.DataFrame,
    ) -> tuple[dict[str, tuple[int, int]], dict[str, int], list[str]]:
        """Build compact split-scoped lookup indices for O(1) entity retrieval.

        Returns:
            Tuple of:
            - ``event_ranges``: ``{entity_id: (start_row, length)}``
            - ``static_index``: ``{entity_id: static_row_index}``
            - ``entity_order``: ordered list of entity IDs (from static table)
        """
        if not events_df.is_empty():
            grouped = (
                events_df.with_row_index("__row")
                .group_by("entity_id", maintain_order=True)
                .agg(pl.first("__row"), pl.len().alias("__len"))
            )
            event_ranges: dict[str, tuple[int, int]] = {
                str(eid): (int(r), int(l))
                for eid, r, l in zip(
                    grouped["entity_id"].cast(pl.Utf8).to_list(),
                    grouped["__row"].to_list(),
                    grouped["__len"].to_list(),
                )
            }
        else:
            event_ranges = {}

        entity_order = static_df["entity_id"].cast(pl.Utf8).to_list()
        static_index = {str(eid): idx for idx, eid in enumerate(entity_order)}

        return event_ranges, static_index, [str(e) for e in entity_order]

    # ------------------------------------------------------------------
    # Private — dataset construction helpers
    # ------------------------------------------------------------------

    def _build_static_table(self, split_df: pl.DataFrame) -> pl.DataFrame:
        static_cols = [c for c in split_df.columns if c != "split"]
        return split_df.select([*static_cols, "split"]).sort("entity_id")

    @staticmethod
    def _event_feature_columns(source: Any) -> list[str]:
        """Return source feature columns that should be encoded per event.

        Static columns are intentionally excluded from event tokenization because
        they belong in the entity-level static payload.
        """
        cols: list[str] = []
        for group in (source.config.categorical_cols, source.config.continuous_cols):
            if not group:
                continue
            cols.extend(cfg.col_name for cfg in group if not cfg.static)
        return cols

    def _empty_events_frame(self, include_token_str: bool = True) -> pl.DataFrame:
        schema: dict[str, pl.DataType] = {
            "entity_id": pl.Utf8,
            "split": pl.Utf8,
            "source_name": pl.Utf8,
            "primary_timestamp": pl.Utf8,
            "primary_time": pl.Int64,
            "token_ids": pl.List(pl.Int64),
        }
        if include_token_str:
            schema["token_str"] = pl.Utf8
        if self.dataset_config.include_after_threshold:
            schema["after_threshold"] = pl.Boolean
        for rule in self.dataset_config.relative_date_features:
            schema[rule.output_column] = pl.Int64 if rule.floor_int else pl.Float64
        return pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in schema.items()})

    # ------------------------------------------------------------------
    # Private — hashing and paths
    # ------------------------------------------------------------------

    def _dataset_hash(self) -> str:
        payload = {
            "dataset_config_hash": self.dataset_config.config_hash(),
            "split_hash": self.split_config.config_hash(),
            "vocab_hash": (self.tokenizer.vocabulary.metadata or {}).get("vocab_hash"),
            "source_hashes": {
                source.name: hashlib.sha256(
                    source.config.model_dump_json().encode()
                ).hexdigest()[:16]
                for source in self.cohort.collection
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _dataset_root(self, dataset_hash: str) -> Path:
        if self.dataset_config.output_dir is not None:
            return Path(self.dataset_config.output_dir) / dataset_hash
        if self.cohort.cache_dir is not None:
            return self.cohort.cache_dir / "datasets" / dataset_hash
        return Path("data/cohorts") / self.cohort.name / "datasets" / dataset_hash

    def _resolve_registry_dir(self, registry_dir: str | Path | None) -> Path:
        if registry_dir is not None:
            return Path(registry_dir)
        if self._loaded_dataset_dir is not None:
            return self._loaded_dataset_dir.parent
        if self.cohort is not None and self.cohort.cache_dir is not None:
            return self.cohort.cache_dir / "datasets"
        raise ValueError(
            "registry_dir is required when cohort cache directory is unavailable."
        )

    @staticmethod
    def _infer_vocab_dir(dataset_dir: Path, vocab_hash: str) -> Path:
        cohort_dir = dataset_dir.parent.parent
        vocab_dir = cohort_dir / "vocabulary" / vocab_hash
        if not vocab_dir.exists():
            raise FileNotFoundError(
                f"Vocabulary directory does not exist: {vocab_dir}"
            )
        return vocab_dir

    @staticmethod
    def _load_vocabulary(vocab_dir: Path) -> Vocabulary:
        vocab_path = vocab_dir / "vocab.parquet"
        metadata_path = vocab_dir / "metadata.json"
        bin_edges_path = vocab_dir / "bin_edges.parquet"

        for path in (vocab_path, metadata_path, bin_edges_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing vocabulary artifact: {path}")

        vocabulary = Vocabulary()
        vocabulary.vocab_df = pl.read_parquet(vocab_path)
        vocabulary.bin_edges_df = pl.read_parquet(bin_edges_path)
        vocabulary.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vocabulary._build_lookup_dicts()
        return vocabulary

    @staticmethod
    def _load_split_config(
        dataset_dir: Path,
        metadata: dict[str, Any],
        entry: DatasetRegistryEntry,
    ) -> CohortConfig:
        split_config = metadata.get("split_config")
        if isinstance(split_config, dict):
            return CohortConfig(**split_config)

        split_hash = entry.split_hash or metadata.get("split_hash")
        if isinstance(split_hash, str):
            split_meta = dataset_dir.parent.parent / "splits" / split_hash / "metadata.json"
            if split_meta.exists():
                payload = json.loads(split_meta.read_text(encoding="utf-8"))
                cfg = payload.get("split_config")
                if isinstance(cfg, dict):
                    return CohortConfig(**cfg)

        raise ValueError(
            "Unable to resolve split configuration for a dataset loaded by name."
        )

    @staticmethod
    def _artifacts_from_dataset_dir(dataset_dir: Path) -> DatasetArtifacts:
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        split_paths: dict[str, Path] = {}
        split_row_counts = metadata.get("split_row_counts", {})
        if not isinstance(split_row_counts, dict):
            raise ValueError(
                f"Dataset metadata at {metadata_path} has invalid split_row_counts."
            )
        for split_name in split_row_counts:
            split_path = dataset_dir / split_name / "part-000.parquet"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split parquet: {split_path}")
            split_paths[split_name] = split_path

        static_path = dataset_dir / "static" / "entities_static.parquet"
        if not static_path.exists():
            raise FileNotFoundError(f"Dataset static table is missing: {static_path}")

        return DatasetArtifacts(
            dataset_dir=dataset_dir,
            metadata_path=metadata_path,
            static_path=static_path,
            split_paths=split_paths,
        )

    # ------------------------------------------------------------------
    # Private — validation and static utilities
    # ------------------------------------------------------------------

    def _require_fitted_tokenizer(self) -> None:
        vocab = self.tokenizer.vocabulary
        if vocab.vocab_df is None:
            raise ValueError(
                "Tokenizer vocabulary must be fitted before building event datasets."
            )
        if vocab.bin_edges_df is None:
            raise ValueError(
                "Vocabulary bin edges are missing — fit the vocabulary first."
            )

    @staticmethod
    def _datetime_offset(event: date, origin: date, unit: str) -> int:
        if unit == "days":
            return (event - origin).days
        if unit == "weeks":
            return (event - origin).days // 7
        if unit == "months":
            return (event.year - origin.year) * 12 + (event.month - origin.month)
        # "years": floor to completed years — subtract 1 if birthday not yet
        # reached in the event year (e.g. born Nov, event in Mar → age - 1).
        years = event.year - origin.year
        if (event.month, event.day) < (origin.month, origin.day):
            years -= 1
        return years

    @staticmethod
    def _coerce_to_date(value: Any) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return date.fromisoformat(value)
        raise TypeError(f"Cannot coerce {type(value).__name__!r} to date.")