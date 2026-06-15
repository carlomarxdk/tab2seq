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
from typing import Any, Iterator

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
    ) -> pl.DataFrame:
        """Build tokenized event rows for one split.

        Each output row represents one event with columns:

        - ``entity_id``, ``split``, ``source_name``
        - ``primary_timestamp`` (ISO string), ``primary_time`` (days since reference)
        - ``token_ids`` (``List[Int64]``) — from :meth:`Tokenizer.encode_frame`
        - ``token_str`` (``Utf8``) — if ``dataset_config.include_token_str``
        - ``after_threshold`` (``Boolean``) — if ``dataset_config.include_after_threshold``
        - Relative date feature columns — per ``dataset_config.relative_date_features``

        Args:
            split_name: Split to build (``'train'``, ``'val'``, ``'test'``).
            force_recompute_splits: Clear caches and recompute splits from scratch.
            split_df: Pre-loaded split DataFrame to avoid redundant I/O.

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
                include_token_str=self.dataset_config.include_token_str,
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
            if self.dataset_config.include_token_str:
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
            pl.concat(source_frames) if source_frames else self._empty_events_frame()
        )

        if self.dataset_config.embed_static_in_events:
            events_df = events_df.join(split_static_df, on="entity_id", how="left")

        events_df = events_df.sort(["entity_id", "primary_timestamp", "source_name"])

        self._split_cache[split_name] = (events_df, split_static_df)
        self._split_entity_index[split_name] = self._build_entity_index(
            events_df, split_static_df
        )
        return events_df

    def build_all_splits(
        self, force_recompute_splits: bool = False
    ) -> dict[str, pl.DataFrame]:
        """Build tokenized event rows for all available splits.

        Args:
            force_recompute_splits: Clear caches and recompute from scratch.

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

        Returns:
            :class:`DatasetArtifacts` with all output paths.
        """
        if self._loaded_from_artifacts:
            raise ValueError(
                "Cannot write_parquet() from a dataset loaded via from_name()."
            )

        splits = self.build_all_splits(force_recompute_splits=force_recompute_splits)
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
        force_recompute_splits: bool = False,
    ) -> dict[str, Any] | None:
        """Return one entity payload with static attributes and event rows.

        Args:
            entity_id: Entity ID to retrieve.
            split: Split name (``'train'``, ``'val'``, ``'test'``).
            force_recompute_splits: Clear caches and recompute from scratch.

        Returns:
            Dict with keys ``entity_id``, ``split``, ``static`` (dict), and
            ``events`` (list of dicts), or ``None`` if not found.
        """
        events_df, static_df = self._get_split_bundle(split, force_recompute_splits)
        event_ranges, static_index, _ = self._split_entity_index[split]
        entity_id = str(entity_id)

        event_slice = event_ranges.get(entity_id)
        events_for_entity = (
            events_df.slice(*event_slice)
            if event_slice is not None
            else events_df.clear()
        )

        static_row_idx = static_index.get(entity_id)
        static_for_entity = (
            static_df.slice(static_row_idx, 1)
            if static_row_idx is not None
            else static_df.clear()
        )

        if events_for_entity.is_empty() and static_for_entity.is_empty():
            return None

        static_payload: dict[str, Any] = {
            "entity_id": entity_id,
            "split": split,
            "token_ids": [],
            "token_str": "",
        }
        if not static_for_entity.is_empty():
            row = static_for_entity.to_dicts()[0]
            static_payload.update(row)

            # Tokenize static attributes into the same vocabulary space.
            feature_keys = sorted(
                key
                for key in static_payload
                if key not in {"entity_id", "split", "token_ids", "token_str"}
            )
            tokens: list[str] = []
            unk_id = self.tokenizer.token2index.get(self.tokenizer.config.unk_token)
            if unk_id is None:
                raise ValueError("Tokenizer vocabulary is missing the UNK special token.")

            token_ids: list[int] = []
            for key in feature_keys:
                value = static_payload.get(key)
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
                token = f"{key}__{value}"
                tokens.append(token)
                token_ids.append(self.tokenizer.token2index.get(token, unk_id))

            static_payload["token_ids"] = token_ids
            static_payload["token_str"] = " ".join(tokens)

        return {
            "entity_id": entity_id,
            "split": split,
            "static": static_payload,
            "events": events_for_entity.to_dicts(),
        }

    def sample_entity_record(
        self,
        split: str,
        seed: int | None = None,
        force_recompute_splits: bool = False,
    ) -> dict[str, Any] | None:
        """Return one randomly sampled entity record from a split.

        Args:
            split: Split name.
            seed: Optional random seed for reproducibility.
            force_recompute_splits: Clear caches and recompute from scratch.

        Returns:
            Entity record dict, or ``None`` if split is empty.
        """
        self._get_split_bundle(split, force_recompute_splits)
        entity_ids = self._entity_order(split)
        if not entity_ids:
            return None
        rng = np.random.default_rng(seed)
        chosen = str(rng.choice(np.array(entity_ids, dtype=object)))
        return self.get_entity_record(chosen, split=split)

    def iter_entity_records(
        self,
        split: str,
        shuffle: bool = False,
        seed: int | None = None,
        force_recompute_splits: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Iterate all entity records in a split.

        Args:
            split: Split name.
            shuffle: Shuffle entity order before iteration.
            seed: Optional random seed for reproducibility.
            force_recompute_splits: Clear caches and recompute from scratch.

        Yields:
            Entity record dicts.
        """
        self._get_split_bundle(split, force_recompute_splits)
        entity_ids = self._entity_order(split)

        if shuffle and entity_ids:
            rng = np.random.default_rng(seed)
            entity_ids = [
                str(v)
                for v in rng.permutation(np.array(entity_ids, dtype=object)).tolist()
            ]

        for entity_id in entity_ids:
            record = self.get_entity_record(str(entity_id), split=split)
            if record is not None:
                yield record

    def next_entity_record(
        self,
        split: str,
        shuffle: bool = False,
        seed: int | None = None,
        reset: bool = False,
        force_recompute_splits: bool = False,
    ) -> dict[str, Any] | None:
        """Return the next entity record in sweep order, or ``None`` when exhausted.

        State is maintained per ``(split, shuffle, seed)`` key.  Call with
        ``reset=True`` to restart iteration from the beginning.

        Args:
            split: Split name.
            shuffle: Shuffle entity order.
            seed: Optional random seed for reproducibility.
            reset: Reset iteration state for this split/shuffle/seed combination.
            force_recompute_splits: Clear caches and recompute from scratch.

        Returns:
            Entity record dict, or ``None`` when all entities have been returned.
        """
        key = (split, shuffle, seed)
        if force_recompute_splits:
            self._split_cache.clear()
            self._next_state.clear()

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
        return self.get_entity_record(entity_id, split=split)

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

    def _empty_events_frame(self) -> pl.DataFrame:
        schema: dict[str, pl.DataType] = {
            "entity_id": pl.Utf8,
            "split": pl.Utf8,
            "source_name": pl.Utf8,
            "primary_timestamp": pl.Utf8,
            "primary_time": pl.Int64,
            "token_ids": pl.List(pl.Int64),
        }
        if self.dataset_config.include_token_str:
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