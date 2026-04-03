"""Polars-based vocabulary builder tied to cohort train split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tab2seq.cohort import Cohort, CohortConfig
from .config import VocabularyConfig


@dataclass
class VocabularyArtifacts:
    """Paths to persisted vocabulary artifacts."""

    directory: Path
    vocab_path: Path
    metadata_path: Path
    bin_edges_path: Path


class Vocabulary:
    """Build and cache token vocabulary using cohort train split entities only."""

    def __init__(
        self,
        config: VocabularyConfig | None = None,
    ) -> None:
        self.config = config or VocabularyConfig()
        self.vocab_df: pl.DataFrame | None = None
        self.bin_edges_df: pl.DataFrame | None = None
        self.metadata: dict[str, Any] | None = None

    def fit_from_cohort_train(
        self,
        cohort: Cohort,
        split_config: CohortConfig | None = None,
        force_recompute: bool = False,
    ) -> pl.DataFrame:
        """Build vocabulary from train split rows and persist to cohort cache."""
        split_cfg = split_config or CohortConfig(
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
        )
        split_df = cohort.build_or_load_splits(split_cfg, force_recompute=force_recompute)

        train_ids = self._resolve_train_ids(split_df)
        vocab_hash = self._vocab_hash(cohort, split_cfg)
        artifacts = self._artifact_paths(cohort, vocab_hash)

        if (
            cohort.use_cache
            and not force_recompute
            and artifacts.vocab_path.exists()
            and artifacts.metadata_path.exists()
            and artifacts.bin_edges_path.exists()
        ):
            self.vocab_df = self._ensure_pretty_token_column(pl.read_parquet(artifacts.vocab_path))
            self.bin_edges_df = pl.read_parquet(artifacts.bin_edges_path)
            self.metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            return self.vocab_df

        token_counts: dict[tuple[str, str, str, str], int] = {}
        bin_edges_rows: list[dict[str, Any]] = []
        sources = self._cohort_sources(cohort)

        for source in sources:
            lf = source.scan().filter(
                pl.col(source.config.id_col).cast(pl.Utf8).is_in(train_ids)
            )
            token_counts = self._collect_categorical_tokens(lf, source.name, source.config.categorical_cols, token_counts)
            token_counts, new_edges = self._collect_continuous_tokens(
                lf,
                source.name,
                source.config.continuous_cols,
                token_counts,
            )
            bin_edges_rows.extend(new_edges)

        rows = self._compose_vocab_rows(token_counts)
        vocab_df = self._ensure_pretty_token_column(pl.DataFrame(rows).sort("token_id"))
        bin_edges_df = pl.DataFrame(bin_edges_rows) if bin_edges_rows else pl.DataFrame(
            {
                "source_name": pl.Series([], dtype=pl.Utf8),
                "column_name": pl.Series([], dtype=pl.Utf8),
                "bin_index": pl.Series([], dtype=pl.Int64),
                "left": pl.Series([], dtype=pl.Float64),
                "right": pl.Series([], dtype=pl.Float64),
            }
        )

        metadata = {
            "cohort_name": cohort.name,
            "vocab_hash": vocab_hash,
            "split_hash": split_cfg.config_hash(),
            "config": {
                "max_vocab_size": self.config.max_vocab_size,
                "min_token_count": self.config.min_token_count,
                "special_tokens": self.config.special_tokens,
            },
            "n_tokens": vocab_df.height,
            "n_train_entities": len(train_ids),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if cohort.use_cache:
            artifacts.directory.mkdir(parents=True, exist_ok=True)
            vocab_df.write_parquet(artifacts.vocab_path)
            bin_edges_df.write_parquet(artifacts.bin_edges_path)
            artifacts.metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

        self.vocab_df = vocab_df
        self.bin_edges_df = bin_edges_df
        self.metadata = metadata
        return vocab_df

    @property
    def token2index(self) -> dict[str, int]:
        """Mapping from token string to token id."""
        if self.vocab_df is None:
            return {}
        return dict(
            zip(
                self.vocab_df.get_column("token").to_list(),
                self.vocab_df.get_column("token_id").to_list(),
            )
        )

    @property
    def index2token(self) -> dict[int, str]:
        """Mapping from token id to token string."""
        if self.vocab_df is None:
            return {}
        return dict(
            zip(
                self.vocab_df.get_column("token_id").to_list(),
                self.vocab_df.get_column("token").to_list(),
            )
        )

    def _resolve_train_ids(self, split_df: pl.DataFrame) -> list[str]:
        split_values = set(split_df.get_column("split").to_list())
        split_name = "train" if "train" in split_values else "all"
        return (
            split_df.filter(pl.col("split") == split_name)
            .get_column("entity_id")
            .cast(pl.Utf8)
            .to_list()
        )

    def _compose_vocab_rows(
        self,
        token_counts: dict[tuple[str, str, str, str], int],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        token_id = 0

        for token in self.config.special_tokens:
            rows.append(
                {
                    "token_id": token_id,
                    "token": token,
                    "pretty_token": token,
                    "category": "special",
                    "source_name": "__special__",
                    "column_name": "__special__",
                    "transform": "identity",
                    "count": -1,
                }
            )
            token_id += 1

        sortable = [
            (key, count)
            for key, count in token_counts.items()
            if count >= self.config.min_token_count
        ]
        sortable.sort(key=lambda x: (x[0][0], x[0][1], x[0][3]))

        for (source_name, column_name, category, token), count in sortable:
            if token_id >= self.config.max_vocab_size:
                break
            rows.append(
                {
                    "token_id": token_id,
                    "token": token,
                    "pretty_token": self._make_pretty_token(token, source_name),
                    "category": category,
                    "source_name": source_name,
                    "column_name": column_name,
                    "transform": category,
                    "count": int(count),
                }
            )
            token_id += 1

        return rows

    def _collect_categorical_tokens(
        self,
        lf: pl.LazyFrame,
        source_name: str,
        categorical_cols: list[Any] | None,
        token_counts: dict[tuple[str, str, str, str], int],
    ) -> dict[tuple[str, str, str, str], int]:
        if not categorical_cols:
            return token_counts

        for col_cfg in categorical_cols:
            col = col_cfg.col_name
            counts = (
                lf.select(pl.col(col).cast(pl.Utf8).alias(col))
                .drop_nulls()
                .group_by(col)
                .len()
                .collect()
            )
            for row in counts.iter_rows(named=True):
                token = f"{source_name}__{col}__{row[col]}"
                key = (source_name, col, "categorical", token)
                token_counts[key] = int(row["len"])
        return token_counts

    def _collect_continuous_tokens(
        self,
        lf: pl.LazyFrame,
        source_name: str,
        continuous_cols: list[Any] | None,
        token_counts: dict[tuple[str, str, str, str], int],
    ) -> tuple[dict[tuple[str, str, str, str], int], list[dict[str, Any]]]:
        if not continuous_cols:
            return token_counts, []

        edge_rows: list[dict[str, Any]] = []

        for col_cfg in continuous_cols:
            col = col_cfg.col_name
            n_bins = col_cfg.n_bins
            values = (
                lf.select(pl.col(col).cast(pl.Float64).alias(col))
                .drop_nulls()
                .collect()
                .get_column(col)
                .to_numpy()
            )
            if values.size == 0:
                continue

            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(values, quantiles)
            edges = np.unique(edges)
            if edges.size < 2:
                continue

            bin_idx = np.searchsorted(edges, values, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, edges.size - 2)
            unique_bins, counts = np.unique(bin_idx, return_counts=True)

            for idx, count in zip(unique_bins.tolist(), counts.tolist()):
                token = f"{source_name}__{col}__BIN_{idx}"
                key = (source_name, col, "continuous_bin", token)
                token_counts[key] = int(count)

            for i in range(edges.size - 1):
                edge_rows.append(
                    {
                        "source_name": source_name,
                        "column_name": col,
                        "bin_index": int(i),
                        "left": float(edges[i]),
                        "right": float(edges[i + 1]),
                    }
                )

        return token_counts, edge_rows

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
        raise TypeError(f"Unsupported datetime value: {type(value).__name__}")

    def _vocab_hash(self, cohort: Cohort, split_cfg: CohortConfig) -> str:
        sources = self._cohort_sources(cohort)
        payload = {
            "vocabulary_config_hash": self._config_hash(),
            "split_hash": split_cfg.config_hash(),
            "source_hashes": {
                source.name: hashlib.sha256(
                    source.config.model_dump_json().encode()
                ).hexdigest()[:16]
                for source in sources
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def _cohort_sources(cohort: Cohort) -> Any:
        if hasattr(cohort, "collection"):
            return cohort.collection
        if hasattr(cohort, "_collection"):
            return cohort._collection
        raise AttributeError("Cohort object does not expose a source collection")

    def _config_hash(self) -> str:
        payload = {
            "max_vocab_size": self.config.max_vocab_size,
            "min_token_count": self.config.min_token_count,
            "special_tokens": self.config.special_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def _make_pretty_token(token: str, source_name: str) -> str:
        prefix = f"{source_name}__"
        if token.startswith(prefix):
            return token[len(prefix) :]
        return token

    @staticmethod
    def _ensure_pretty_token_column(vocab_df: pl.DataFrame) -> pl.DataFrame:
        if "pretty_token" in vocab_df.columns:
            return vocab_df
        return vocab_df.with_columns(
            pl.when(pl.col("source_name") == "__special__")
            .then(pl.col("token"))
            .otherwise(pl.col("token").str.split_exact("__", 1).struct.field("field_1"))
            .alias("pretty_token")
        )

    def _artifact_paths(self, cohort: Cohort, vocab_hash: str) -> VocabularyArtifacts:
        base = cohort.vocabulary_cache_dir(vocab_hash)
        if base is None:
            raise ValueError(
                "Vocabulary caching requires cohort cache to be enabled with a valid cache_dir."
            )
        return VocabularyArtifacts(
            directory=base,
            vocab_path=base / "vocab.parquet",
            metadata_path=base / "metadata.json",
            bin_edges_path=base / "bin_edges.parquet",
        )
