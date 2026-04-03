"""Build merged tokenized event datasets from cohort splits and fitted vocabulary."""

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
from tab2seq.tokenization import Vocabulary


@dataclass
class DatasetArtifacts:
    """Output paths for persisted event dataset artifacts.
    Attributes:
        dataset_dir: Root directory containing all dataset artifacts.
        metadata_path: Path to JSON file containing dataset metadata (e.g. schema, row counts, config hashes).
        static_path: Path to Parquet file containing static entity attributes.
        split_paths: Dictionary mapping split names to Parquet file paths containing event rows for each split.
    """

    dataset_dir: Path
    metadata_path: Path
    static_path: Path
    split_paths: dict[str, Path]


class EventDataset:
    """Construct split-aware tokenized event rows and persist to Parquet."""

    def __init__(
        self,
        cohort: Cohort,
        vocabulary: Vocabulary,
        split_config: CohortConfig | None = None,
        dataset_config: EventDatasetConfig | None = None,
    ) -> None:
        self.cohort = cohort
        self.vocabulary = vocabulary
        self.split_config = split_config or CohortConfig(
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
        )
        self.dataset_config = dataset_config or EventDatasetConfig()

        self._require_fitted_vocabulary()
        self._token2id = self.vocabulary.token2index
        self._unk_id = self._token2id.get("[UNK]", -1)
        self._continuous_edges = self._build_continuous_edges_map()
        self._split_cache: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
        self._split_entity_index: dict[
            str,
            tuple[dict[str, tuple[int, int]], dict[str, int], list[str]],
        ] = {}
        self._next_state: dict[tuple[str, bool, int | None], dict[str, Any]] = {}

    def build_split(
        self,
        split_name: str,
        force_recompute_splits: bool = False,
        split_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Build tokenized event dataset rows for one split.
        Args:
            split_name: Name of the split to build (e.g. 'train', 'val', 'test').
            force_recompute_splits: If True, forces recomputation of splits and clears related caches
            split_df: Optional pre-loaded split DataFrame to avoid redundant I/O.
        Returns:
            Polars DataFrame with one row per event, containing tokenized features and static attributes.
        """
        if force_recompute_splits:
            self._split_cache.clear()
            self._split_entity_index.clear()
            self._next_state.clear()

        if split_df is None:
            split_df = self.cohort.build_or_load_splits(
                self.split_config,
                force_recompute=force_recompute_splits,
            )
        available = set(split_df.get_column("split").to_list())
        if split_name not in available:
            raise ValueError(
                f"split '{split_name}' not found. available splits: {sorted(available)}"
            )

        split_ids = set(
            split_df.filter(pl.col("split") == split_name)
            .get_column("entity_id")
            .cast(pl.Utf8)
            .to_list()
        )
        static_df = self._build_static_table(split_df)
        split_static_df = static_df.filter(pl.col("split") == split_name)

        # Materialize O(1) static lookup for _apply_relative_rules using only
        # the current split to avoid converting the full static table to Python.
        static_lookup: dict[str, dict] = {
            str(d["entity_id"]): d for d in split_static_df.to_dicts()
        }

        ref_date = date.fromisoformat(self.dataset_config.reference_date)
        threshold_date = (
            date.fromisoformat(self.dataset_config.threshold_date)
            if self.dataset_config.include_after_threshold
            else None
        )

        source_frames: list[pl.DataFrame] = []
        for source in self.cohort.collection:
            primary_ts = source.config.primary_timestamp
            if primary_ts is None:
                continue

            source_df = (
                source.scan()
                .filter(pl.col(source.config.id_col).cast(pl.Utf8).is_in(split_ids))
                .collect()
            )
            if source_df.height == 0:
                continue

            ts_col = primary_ts.col_name
            # Coerce timestamp column once instead of per-row
            source_df = source_df.with_columns(pl.col(ts_col).cast(pl.Date))

            src_name = source.name
            id_col = source.config.id_col

            # --- Vectorized tokenization ---
            token_str_cols: list[str] = []
            token_id_cols: list[str] = []
            exprs: list[pl.Expr] = []

            # Categorical tokens
            for cat_cfg in source.config.categorical_cols or []:
                col_name = cat_cfg.col_name
                tok_alias = f"__tok_cat_{col_name}"
                id_alias = f"__tid_cat_{col_name}"
                token_str_cols.append(tok_alias)
                token_id_cols.append(id_alias)
                prefix = f"{src_name}__{col_name}__"
                exprs.append(
                    pl.when(pl.col(col_name).is_not_null())
                    .then(pl.lit(prefix) + pl.col(col_name).cast(pl.Utf8))
                    .otherwise(pl.lit(None, dtype=pl.Utf8))
                    .alias(tok_alias)
                )

            # Continuous tokens (vectorized np.searchsorted)
            for cont_cfg in source.config.continuous_cols or []:
                col_name = cont_cfg.col_name
                edges = self._continuous_edges.get((src_name, col_name))
                if edges is None or edges.size < 2:
                    continue

                tok_alias = f"__tok_cont_{col_name}"
                id_alias = f"__tid_cont_{col_name}"
                token_str_cols.append(tok_alias)
                token_id_cols.append(id_alias)

                raw_vals = source_df.get_column(col_name)
                vals_filled = raw_vals.fill_null(0.0).to_numpy()
                bin_indices = np.clip(
                    np.searchsorted(edges, vals_filled, side="right") - 1,
                    0,
                    edges.size - 2,
                )
                prefix = f"{src_name}__{col_name}__BIN_"
                tok_strs = np.array([f"{prefix}{i}" for i in bin_indices], dtype=object)
                tok_series = pl.Series(tok_alias, tok_strs, dtype=pl.Utf8)
                tok_expr = (
                    pl.when(pl.col(col_name).is_not_null())
                    .then(pl.lit(tok_series))
                    .otherwise(pl.lit(None, dtype=pl.Utf8))
                    .alias(tok_alias)
                )
                exprs.append(tok_expr)

            if exprs:
                source_df = source_df.with_columns(exprs)

            # Map token strings to IDs
            t2id = self._token2id
            unk = self._unk_id
            id_exprs: list[pl.Expr] = []
            for tok_col, id_col_name in zip(token_str_cols, token_id_cols):
                id_exprs.append(
                    pl.when(pl.col(tok_col).is_not_null())
                    .then(pl.col(tok_col).replace_strict(t2id, default=unk).cast(pl.Int64))
                    .otherwise(pl.lit(None, dtype=pl.Int64))
                    .alias(id_col_name)
                )
            if id_exprs:
                source_df = source_df.with_columns(id_exprs)

            # Build token_ids list column (drop nulls per row)
            if token_id_cols:
                source_df = source_df.with_columns(
                    pl.concat_list([pl.col(c) for c in token_id_cols])
                    .list.eval(pl.element().drop_nulls())
                    .alias("token_ids")
                )
            else:
                source_df = source_df.with_columns(
                    pl.lit(None, dtype=pl.List(pl.Int64)).alias("token_ids")
                )

            # Build token_str column
            if self.dataset_config.include_token_str:
                if token_str_cols:
                    source_df = source_df.with_columns(
                        pl.concat_list([pl.col(c) for c in token_str_cols])
                        .list.eval(pl.element().drop_nulls())
                        .list.join(" ")
                        .alias("token_str")
                    )
                else:
                    source_df = source_df.with_columns(
                        pl.lit("", dtype=pl.Utf8).alias("token_str")
                    )

            # Build scalar columns
            source_df = source_df.with_columns(
                pl.col(id_col).cast(pl.Utf8).alias("entity_id"),
                pl.lit(split_name).alias("split"),
                pl.lit(src_name).alias("source_name"),
                pl.col(ts_col).cast(pl.Utf8).alias("primary_timestamp"),
                (pl.col(ts_col) - pl.lit(ref_date)).dt.total_days().cast(pl.Int64).alias("primary_time"),
            )

            if self.dataset_config.include_after_threshold:
                source_df = source_df.with_columns(
                    (pl.col(ts_col) >= pl.lit(threshold_date)).alias("after_threshold")
                )

            # Select output columns
            out_cols = [
                "entity_id", "split", "source_name",
                "primary_timestamp", "primary_time", "token_ids",
            ]
            if self.dataset_config.include_token_str:
                out_cols.append("token_str")
            if self.dataset_config.include_after_threshold:
                out_cols.append("after_threshold")

            source_out = source_df.select(out_cols)

            # Apply relative date rules (O(1) dict lookup per row)
            if self.dataset_config.relative_date_features:
                rule_data: dict[str, list] = {
                    rule.output_column: [] for rule in self.dataset_config.relative_date_features
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
                rule_cols = []
                for rule in self.dataset_config.relative_date_features:
                    dtype = pl.Int64 if rule.floor_int else pl.Float64
                    rule_cols.append(
                        pl.Series(rule.output_column, rule_data[rule.output_column], dtype=dtype)
                    )
                source_out = source_out.hstack(rule_cols)

            source_frames.append(source_out)

        events_df = pl.concat(source_frames) if source_frames else self._empty_events_frame()

        if self.dataset_config.embed_static_in_events:
            events_df = events_df.join(split_static_df, on="entity_id", how="left")

        events_df = events_df.sort(["entity_id", "primary_timestamp", "source_name"])
        self._split_cache[split_name] = (events_df, split_static_df)
        self._split_entity_index[split_name] = self._build_entity_index(events_df, split_static_df)
        return events_df

    def build_all_splits(self, force_recompute_splits: bool = False) -> dict[str, pl.DataFrame]:
        """Build tokenized event rows for all available splits.
        Args:
            force_recompute_splits: If True, forces recomputation of splits and clears related caches
        Returns:
            Dictionary mapping split names to their corresponding tokenized event DataFrames."""
        split_df = self.cohort.build_or_load_splits(
            self.split_config,
            force_recompute=force_recompute_splits,
        )
        split_names = sorted(set(split_df.get_column("split").to_list()))
        return {
            split_name: self.build_split(
                split_name,
                force_recompute_splits=force_recompute_splits,
                split_df=split_df,
            )
            for split_name in split_names
        }

    def build_static_table(self, force_recompute_splits: bool = False) -> pl.DataFrame:
        """Build static entity table (entity_id + split + static columns).
        Args:
            force_recompute_splits: If True, forces recomputation of splits and clears related caches
        Returns:
            Polars DataFrame with one row per entity, containing static attributes and split assignment.
        """
        split_df = self.cohort.build_or_load_splits(
            self.split_config,
            force_recompute=force_recompute_splits,
        )
        return self._build_static_table(split_df)

    def write_parquet(self, force_recompute_splits: bool = False) -> DatasetArtifacts:
        """Persist split event datasets and static table as Parquet artifacts.
        Args:
            force_recompute_splits: If True, forces recomputation of splits and clears related caches
        Returns:
            DatasetArtifacts containing paths to the persisted dataset files and metadata.
        """
        splits = self.build_all_splits(force_recompute_splits=force_recompute_splits)
        static_df = self.build_static_table(force_recompute_splits=force_recompute_splits)

        dataset_hash = self._dataset_hash()
        root = self._dataset_root(dataset_hash)
        root.mkdir(parents=True, exist_ok=True)

        split_paths: dict[str, Path] = {}
        for split_name, frame in splits.items():
            split_dir = root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_path = split_dir / "part-000.parquet"
            frame.write_parquet(split_path)
            split_paths[split_name] = split_path

        static_dir = root / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        static_path = static_dir / "entities_static.parquet"
        static_df.write_parquet(static_path)

        metadata = {
            "dataset_hash": dataset_hash,
            "cohort_name": self.cohort.name,
            "split_hash": self.split_config.config_hash(),
            "vocab_hash": (self.vocabulary.metadata or {}).get("vocab_hash"),
            "dataset_config": self.dataset_config.model_dump(mode="json"),
            "event_schema": {k: str(v) for k, v in splits[next(iter(splits))].schema.items()} if splits else {},
            "static_schema": {k: str(v) for k, v in static_df.schema.items()},
            "split_row_counts": {split: frame.height for split, frame in splits.items()},
        }

        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return DatasetArtifacts(
            dataset_dir=root,
            metadata_path=metadata_path,
            static_path=static_path,
            split_paths=split_paths,
        )

    def get_entity_record(
        self,
        entity_id: str,
        split: str,
        force_recompute_splits: bool = False,
    ) -> dict[str, Any] | None:
        """Return one entity payload containing static attributes and event rows.
        Args:
            entity_id: ID of the entity to retrieve (must be present in the split).
            split: Name of the split to retrieve from (e.g. 'train', 'val', 'test').
            force_recompute_splits: If True, forces recomputation of splits and clears related caches
        Returns:
            Dictionary with keys 'entity_id', 'split', 'static' (dict of static attributes), and 'events' (list of event dicts), or None if entity not found.
        """
        events_df, static_df = self._get_split_bundle(split, force_recompute_splits)
        events_index, static_index, _ = self._split_entity_index[split]

        entity_id = str(entity_id)
        event_slice = events_index.get(entity_id)
        if event_slice is None:
            events_for_entity = events_df.clear()
        else:
            start, length = event_slice
            events_for_entity = events_df.slice(start, length)

        static_row_idx = static_index.get(entity_id)
        if static_row_idx is None:
            static_for_entity = static_df.clear()
        else:
            static_for_entity = static_df.slice(static_row_idx, 1)

        if events_for_entity.height == 0 and static_for_entity.height == 0:
            return None

        static_payload: dict[str, Any] = {}
        if static_for_entity.height > 0:
            static_payload = static_for_entity.drop(["entity_id", "split"]).to_dicts()[0]

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
        """Sample one entity record from a split.
        Args:
            split: Name of the split to sample from (e.g. 'train', 'val', 'test').
            seed: Optional random seed for reproducibility.
            force_recompute_splits: If True, forces recomputation of splits and clears related caches.
        Returns:
            Dictionary with keys 'entity_id', 'split', 'static' (dict of static attributes), and 'events' (list of event dicts), or None if no entities are available.
        """
        self._get_split_bundle(split, force_recompute_splits)
        entity_ids = self._entity_order(split)
        if not entity_ids:
            return None

        rng = np.random.default_rng(seed)
        chosen = str(rng.choice(np.array(entity_ids, dtype=object), size=1)[0])
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
            split: Name of the split to iterate over (e.g. 'train', 'val', 'test').
            shuffle: Whether to shuffle the order of entities before iteration.
            seed: Optional random seed for reproducibility when shuffling.
            force_recompute_splits: If True, forces recomputation of splits and clears related caches.
        Yields:
            Dictionaries with keys 'entity_id', 'split',
            'static' (dict of static attributes), and 'events' (list of event dicts) for each entity in the split.
        """
        self._get_split_bundle(split, force_recompute_splits)
        entity_ids = self._entity_order(split)

        if shuffle and entity_ids:
            rng = np.random.default_rng(seed)
            entity_ids = [str(v) for v in rng.permutation(np.array(entity_ids, dtype=object)).tolist()]

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
        """Return next entity record in sweep order for a split, or None when exhausted.
        Args:
            split: Name of the split to iterate over (e.g. 'train', 'val', 'test').
            shuffle: Whether to shuffle the order of entities before iteration.
            seed: Optional random seed for reproducibility when shuffling.
            reset: If True, resets the iteration state for the split.
            force_recompute_splits: If True, forces recomputation of splits and clears related caches.
        Returns:
            Dictionary with keys 'entity_id', 'split', 'static' (dict of static attributes), and 'events' (list of event dicts), or None if no entities are available.
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
                    str(v) for v in rng.permutation(np.array(entity_ids, dtype=object)).tolist()
                ]
            self._next_state[key] = {"order": entity_ids, "index": 0}

        state = self._next_state[key]
        index = int(state["index"])
        order = state["order"]
        if index >= len(order):
            return None

        entity_id = str(order[index])
        state["index"] = index + 1
        return self.get_entity_record(entity_id, split=split)

    def _get_split_bundle(
        self,
        split: str,
        force_recompute_splits: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if force_recompute_splits:
            self._split_cache.pop(split, None)
            self._split_entity_index.pop(split, None)
        if split not in self._split_cache:
            _ = self.build_split(split, force_recompute_splits=force_recompute_splits)
        if split not in self._split_entity_index:
            events_df, static_df = self._split_cache[split]
            self._split_entity_index[split] = self._build_entity_index(events_df, static_df)
        return self._split_cache[split]

    def _entity_order(self, split: str) -> list[str]:
        if split not in self._split_entity_index:
            _ = self._get_split_bundle(split)
        return self._split_entity_index[split][2]

    def _build_entity_index(
        self,
        events_df: pl.DataFrame,
        static_df: pl.DataFrame,
    ) -> tuple[dict[str, tuple[int, int]], dict[str, int], list[str]]:
        """Build compact split-scoped lookup indices for entity retrieval."""
        static_index: dict[str, int] = {}

        # Vectorized run-length encoding via Polars group_by
        if events_df.height > 0:
            grouped = (
                events_df.with_row_index("__row")
                .group_by("entity_id", maintain_order=True)
                .agg(pl.first("__row"), pl.len().alias("__len"))
            )
            eids = grouped.get_column("entity_id").cast(pl.Utf8).to_list()
            rows = grouped.get_column("__row").to_list()
            lens = grouped.get_column("__len").to_list()
            event_ranges: dict[str, tuple[int, int]] = {
                str(eid): (int(r), int(l)) for eid, r, l in zip(eids, rows, lens)
            }
        else:
            event_ranges = {}

        static_entity_ids = static_df.get_column("entity_id").cast(pl.Utf8).to_list()
        entity_order = [str(entity_id) for entity_id in static_entity_ids]
        for idx, entity_id in enumerate(entity_order):
            static_index[entity_id] = idx

        return event_ranges, static_index, entity_order

    def _build_static_table(self, split_df: pl.DataFrame) -> pl.DataFrame:
        static_cols = [c for c in split_df.columns if c not in {"split"}]
        return split_df.select([*static_cols, "split"]).sort("entity_id")

    def _tokenize_row(self, source_name: str, source_config: Any, row: dict[str, Any]) -> list[str]:
        tokens: list[str] = []

        for cat_cfg in source_config.categorical_cols or []:
            value = row.get(cat_cfg.col_name)
            if value is None:
                continue
            tokens.append(f"{source_name}__{cat_cfg.col_name}__{value}")

        for cont_cfg in source_config.continuous_cols or []:
            value = row.get(cont_cfg.col_name)
            if value is None:
                continue
            token = self._continuous_token(source_name, cont_cfg.col_name, float(value))
            if token is not None:
                tokens.append(token)

        return tokens

    def _continuous_token(self, source_name: str, col_name: str, value: float) -> str | None:
        key = (source_name, col_name)
        edges = self._continuous_edges.get(key)
        if edges is None or edges.size < 2:
            return None
        bin_idx = np.searchsorted(edges, value, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, edges.size - 2))
        return f"{source_name}__{col_name}__BIN_{bin_idx}"

    def _apply_relative_rules(
        self,
        record: dict[str, Any],
        static_lookup: dict[str, dict],
        entity_id: str,
        event_date: date,
    ) -> None:
        if not self.dataset_config.relative_date_features:
            return

        entity_static = static_lookup.get(entity_id)
        if entity_static is None:
            for rule in self.dataset_config.relative_date_features:
                record[rule.output_column] = None
            return

        for rule in self.dataset_config.relative_date_features:
            raw = entity_static.get(rule.source_static_column)
            if raw is None:
                record[rule.output_column] = None
                continue

            static_date = self._coerce_to_date(raw)
            val = self._datetime_offset(event_date, static_date, rule.unit)
            record[rule.output_column] = int(val) if rule.floor_int else float(val)

    def _build_continuous_edges_map(self) -> dict[tuple[str, str], np.ndarray]:
        edges: dict[tuple[str, str], np.ndarray] = {}
        if self.vocabulary.bin_edges_df is None or self.vocabulary.bin_edges_df.height == 0:
            return edges

        grouped = self.vocabulary.bin_edges_df.sort(["source_name", "column_name", "bin_index"])
        for (source_name, col_name), group in grouped.group_by(["source_name", "column_name"]):
            left = group.get_column("left").to_list()
            right = group.get_column("right").to_list()
            boundaries = np.array([left[0], *right], dtype=float)
            edges[(source_name, col_name)] = boundaries
        return edges

    def _days_since_reference(self, value: date) -> int:
        ref = date.fromisoformat(self.dataset_config.reference_date)
        return (value - ref).days

    @staticmethod
    def _datetime_offset(event: date, origin: date, unit: str) -> int:
        if unit == "days":
            return (event - origin).days
        if unit == "weeks":
            return (event - origin).days // 7
        if unit == "months":
            return (event.year - origin.year) * 12 + (event.month - origin.month)
        return event.year - origin.year

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
                try:
                    return date.fromisoformat(value)
                except ValueError as e:
                    raise TypeError(f"Could not parse date from {value!r}") from e

    def _dataset_hash(self) -> str:
        payload = {
            "dataset_config_hash": self.dataset_config.config_hash(),
            "split_hash": self.split_config.config_hash(),
            "vocab_hash": (self.vocabulary.metadata or {}).get("vocab_hash"),
            "source_hashes": {
                source.name: hashlib.sha256(
                    source.config.model_dump_json().encode()
                ).hexdigest()[:16]
                for source in self.cohort.collection
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _dataset_root(self, dataset_hash: str) -> Path:
        if self.dataset_config.output_dir is not None:
            return Path(self.dataset_config.output_dir) / dataset_hash
        if self.cohort.cache_dir is not None:
            return self.cohort.cache_dir / "datasets" / dataset_hash
        return Path("data/cohorts") / self.cohort.name / "datasets" / dataset_hash

    def _require_fitted_vocabulary(self) -> None:
        if self.vocabulary.vocab_df is None:
            raise ValueError("Vocabulary must be fitted before building event datasets.")
        if self.vocabulary.bin_edges_df is None:
            raise ValueError("Vocabulary bin edges are missing. Fit vocabulary first.")

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
            schema[rule.output_column] = pl.Float64 if not rule.floor_int else pl.Int64
        return pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in schema.items()})
