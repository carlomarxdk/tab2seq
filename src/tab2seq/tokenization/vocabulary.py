"""Polars-based vocabulary builder tied to cohort train split.

Token naming convention (``__`` as delimiter, source name never contains ``__``):

- Categorical:  ``{source_name}__{col}__{value}``
- Continuous:   ``{source_name}__{col}__BIN_{bin_index}``
- Special:      ``[PAD]``, ``[UNK]``, ``[CLS]``, ``[SEP]``, ``[MASK]``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    """Paths to persisted vocabulary artifacts on disk."""

    directory: Path
    vocab_path: Path
    metadata_path: Path
    bin_edges_path: Path


class Vocabulary:
    """Build and cache a token vocabulary from cohort train-split data.

    The vocabulary is fitted **exclusively on train-split entities** to prevent
    information leakage from validation/test splits into the token representation.
    Continuous columns are quantile-binned; bin edges are stored alongside the
    vocabulary so the :class:`~tab2seq.tokenizer.tokenizer.Tokenizer` can apply
    identical binning at encode time.

    Attributes:
        config: Vocabulary configuration.
        vocab_df: DataFrame with schema
            ``[token_id, token, pretty_token, category, source_name,
            column_name, transform, count]``.
        bin_edges_df: DataFrame with schema
            ``[source_name, column_name, bin_index, left, right]``.
        metadata: Build metadata dict (hashes, counts, timestamp).

    Example::

        vocab = Vocabulary(VocabularyConfig(max_vocab_size=50_000, min_token_count=5))
        vocab_df = vocab.fit_from_cohort_train(cohort)

        # Inspect
        print(vocab.vocab_df.head())
        print(vocab.column_categories("hospital"))
        # → {"diagnosis": "categorical", "age_at_event": "continuous_bin", ...}

        edges = vocab.bin_edges_for("hospital", "age_at_event")
        # → np.ndarray([18.0, 35.2, 51.7, ..., 97.4])
    """

    def __init__(self, config: VocabularyConfig | None = None) -> None:
        self.config = config or VocabularyConfig()
        self.vocab_df: pl.DataFrame | None = None
        self.bin_edges_df: pl.DataFrame | None = None
        self.metadata: dict[str, Any] | None = None
        self.token2index: dict[str, int] = {}
        self.index2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_from_cohort_train(
        self,
        cohort: Cohort,
        split_config: CohortConfig | None = None,
        force_recompute: bool = False,
    ) -> pl.DataFrame:
        """Build the vocabulary from train-split rows and persist to cache.

        Args:
            cohort: Cohort with source collection and optional cache directory.
            split_config: Train/val/test fractions. Defaults to 70/15/15.
            force_recompute: Ignore cached artifacts and rebuild from scratch.

        Returns:
            ``vocab_df`` DataFrame (also stored as ``self.vocab_df``).
        """
        split_cfg = split_config or CohortConfig(
            train_frac=0.7, val_frac=0.15, test_frac=0.15
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
            self.vocab_df = self._ensure_pretty_token_column(
                pl.read_parquet(artifacts.vocab_path)
            )
            self.bin_edges_df = pl.read_parquet(artifacts.bin_edges_path)
            self.metadata = json.loads(
                artifacts.metadata_path.read_text(encoding="utf-8")
            )
            self._build_lookup_dicts()
            return self.vocab_df

        token_counts: dict[tuple[str, str, str, str, str], int] = {}
        bin_edges_rows: list[dict[str, Any]] = []

        for source in cohort.collection:
            lf = source.scan().filter(
                pl.col(source.config.id_col).cast(pl.Utf8).is_in(train_ids)
            )
            token_counts = self._collect_categorical_tokens(
                lf,
                source.name,
                source.config.id_col,
                source.config.categorical_cols,
                token_counts,
            )
            token_counts, new_edges = self._collect_continuous_tokens(
                lf,
                source.name,
                source.config.id_col,
                source.config.continuous_cols,
                token_counts,
            )
            bin_edges_rows.extend(new_edges)

        vocab_df = self._ensure_pretty_token_column(
            pl.DataFrame(self._compose_vocab_rows(token_counts)).sort("token_id")
        )
        bin_edges_df = (
            pl.DataFrame(bin_edges_rows)
            if bin_edges_rows
            else pl.DataFrame(
                {
                    "source_name": pl.Series([], dtype=pl.Utf8),
                    "column_name": pl.Series([], dtype=pl.Utf8),
                    "bin_index": pl.Series([], dtype=pl.Int64),
                    "left": pl.Series([], dtype=pl.Float64),
                    "right": pl.Series([], dtype=pl.Float64),
                }
            )
        )
        metadata = {
            "cohort_name": cohort.name,
            "vocab_hash": vocab_hash,
            "split_hash": split_cfg.config_hash(),
            "config": {
                "max_vocab_size": self.config.max_vocab_size,
                "min_token_count": self.config.min_token_count,
                "count_mode": self.config.count_mode,
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
        self._build_lookup_dicts()
        return vocab_df

    def _build_lookup_dicts(self) -> None:
        """Populate token2index and index2token from the current vocab_df."""
        if self.vocab_df is None:
            self.token2index = {}
            self.index2token = {}
            return
        tokens = self.vocab_df["token"].to_list()
        ids = self.vocab_df["token_id"].to_list()
        self.token2index = dict(zip(tokens, ids))
        self.index2token = dict(zip(ids, tokens))

    def column_categories(self, source_name: str) -> dict[str, str]:
        """Return ``{column_name: category}`` for all feature columns in a source.

        Categories are ``"categorical"`` or ``"continuous_bin"``.
        Special tokens (``source_name == "__special__"``) are excluded.

        This is the primary way :class:`~tab2seq.tokenizer.tokenizer.Tokenizer`
        discovers which columns are continuous vs. categorical for a given source.

        Args:
            source_name: Source name as registered in the cohort.

        Returns:
            Dict mapping column name to its token category string.
            Empty dict if vocabulary has not been fitted yet.
        """
        if self.vocab_df is None:
            return {}
        sub = (
            self.vocab_df
            .filter(pl.col("source_name") == source_name)
            .select(["column_name", "category"])
            .unique()
        )
        return dict(zip(sub["column_name"].to_list(), sub["category"].to_list()))

    def column_prefixes(self, source_name: str) -> dict[str, str]:
        """Return ``{column_name: prefix}`` for all feature columns in a source.

        Args:
            source_name: Source name as registered in the cohort.

        Returns:
            Dict mapping column name to its configured token prefix string.
            Empty dict if vocabulary has not been fitted yet.
        """
        if self.vocab_df is None or "prefix" not in self.vocab_df.columns:
            return {}
        sub = (
            self.vocab_df
            .filter(
                (pl.col("source_name") == source_name)
                & (pl.col("source_name") != "__special__")
            )
            .select(["column_name", "prefix"])
            .unique()
        )
        return dict(zip(sub["column_name"].to_list(), sub["prefix"].to_list()))

    def bin_edges_for(self, source_name: str, col_name: str) -> np.ndarray | None:
        """Return bin edges for a continuous column as a sorted 1-D array.

        The array has shape ``(n_bins + 1,)`` and is compatible with
        ``np.searchsorted(edges, value, side='right') - 1`` to reproduce
        the same bin assignments made during :meth:`fit_from_cohort_train`.

        Args:
            source_name: Source name as registered in the cohort.
            col_name: Column name of the continuous variable.

        Returns:
            Edge array of shape ``(n_bins + 1,)``, or ``None`` if no bin
            edges exist for this source/column combination.
        """
        if self.bin_edges_df is None or self.bin_edges_df.is_empty():
            return None
        sub = (
            self.bin_edges_df
            .filter(
                (pl.col("source_name") == source_name)
                & (pl.col("column_name") == col_name)
            )
            .sort("bin_index")
        )
        if sub.is_empty():
            return None
        # Reconstruct full edge array from stored (left, right) pairs.
        # Each bin i has left=edges[i], right=edges[i+1], so the full
        # array is lefts + [right of last bin].
        lefts = sub["left"].to_numpy()
        right_last = float(sub["right"][-1])
        return np.append(lefts, right_last)

    # ------------------------------------------------------------------
    # Private helpers — vocabulary construction
    # ------------------------------------------------------------------

    def _compose_vocab_rows(
        self,
        token_counts: dict[tuple[str, str, str, str, str], int],
    ) -> list[dict[str, Any]]:
        """Assemble vocab rows from token counts, prepending special tokens."""
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
                    "prefix": "__special__",
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
        # Sort by (source, column, token) for deterministic ordering.
        sortable.sort(key=lambda x: (x[0][0], x[0][1], x[0][4]))

        for (source_name, column_name, prefix, category, token), count in sortable:
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
                    "prefix": prefix,
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
        id_col: str,
        categorical_cols: list[Any] | None,
        token_counts: dict[tuple[str, str, str, str, str], int],
    ) -> dict[tuple[str, str, str, str, str], int]:
        """Count occurrences of each categorical token in the train split."""
        if not categorical_cols:
            return token_counts

        for col_cfg in categorical_cols:
            col = col_cfg.col_name
            prefix = col_cfg.prefix
            if self.config.count_mode == "entity_unique":
                counts = (
                    lf.select(
                        [
                            pl.col(id_col).cast(pl.Utf8).alias(id_col),
                            pl.col(col).cast(pl.Utf8).alias(col),
                        ]
                    )
                    .drop_nulls()
                    .unique(subset=[id_col, col])
                    .group_by(col)
                    .len()
                    .collect()
                )
            else:
                counts = (
                    lf.select(pl.col(col).cast(pl.Utf8).alias(col))
                    .drop_nulls()
                    .group_by(col)
                    .len()
                    .collect()
                )
            for row in counts.iter_rows(named=True):
                token = f"{source_name}__{prefix}__{row[col]}"
                token_counts[(source_name, col, prefix, "categorical", token)] = int(row["len"])

        return token_counts

    def _collect_continuous_tokens(
        self,
        lf: pl.LazyFrame,
        source_name: str,
        id_col: str,
        continuous_cols: list[Any] | None,
        token_counts: dict[tuple[str, str, str, str, str], int],
    ) -> tuple[dict[tuple[str, str, str, str, str], int], list[dict[str, Any]]]:
        """Quantile-bin continuous columns, count bin occupancy, store edges.

        Bin edges are computed from train-split quantiles only (leakage-safe).
        ``np.unique`` is applied to edges to handle degenerate distributions
        (e.g. many identical values compressing the quantile grid).
        """
        if not continuous_cols:
            return token_counts, []

        edge_rows: list[dict[str, Any]] = []

        for col_cfg in continuous_cols:
            col = col_cfg.col_name
            prefix = col_cfg.prefix
            n_bins = col_cfg.n_bins

            selected = (
                lf.select(
                    [
                        pl.col(id_col).cast(pl.Utf8).alias(id_col),
                        pl.col(col).cast(pl.Float64).alias(col),
                    ]
                )
                .drop_nulls()
                .collect()
            )
            values = selected[col].to_numpy()
            if values.size == 0:
                continue

            edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
            if edges.size < 2:
                continue

            bin_idx = np.clip(
                np.searchsorted(edges, values, side="right") - 1,
                0,
                edges.size - 2,
            )
            if self.config.count_mode == "entity_unique":
                per_event = pl.DataFrame(
                    {
                        id_col: selected[id_col],
                        "bin_idx": pl.Series(bin_idx.tolist(), dtype=pl.Int64),
                    }
                )
                counts_df = (
                    per_event
                    .unique(subset=[id_col, "bin_idx"])
                    .group_by("bin_idx")
                    .len()
                    .sort("bin_idx")
                )
                count_pairs = zip(
                    counts_df["bin_idx"].to_list(),
                    counts_df["len"].to_list(),
                )
            else:
                count_pairs = zip(*np.unique(bin_idx, return_counts=True))

            for idx, count in count_pairs:
                token = f"{source_name}__{prefix}__BIN_{int(idx)}"
                token_counts[(source_name, col, prefix, "continuous_bin", token)] = int(count)

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

    # ------------------------------------------------------------------
    # Private helpers — hashing and paths
    # ------------------------------------------------------------------

    def _resolve_train_ids(self, split_df: pl.DataFrame) -> list[str]:
        split_values = set(split_df["split"].to_list())
        split_name = "train" if "train" in split_values else "all"
        return (
            split_df
            .filter(pl.col("split") == split_name)["entity_id"]
            .cast(pl.Utf8)
            .to_list()
        )

    def _vocab_hash(self, cohort: Cohort, split_cfg: CohortConfig) -> str:
        sources = cohort.collection
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
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _config_hash(self) -> str:
        payload = {
            "max_vocab_size": self.config.max_vocab_size,
            "min_token_count": self.config.min_token_count,
            "count_mode": self.config.count_mode,
            "special_tokens": self.config.special_tokens,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

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

    # ------------------------------------------------------------------
    # Private helpers — static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pretty_token(token: str, source_name: str) -> str:
        """Strip the ``{source_name}__`` prefix for human-readable display."""
        prefix = f"{source_name}__"
        return token[len(prefix):] if token.startswith(prefix) else token

    @staticmethod
    def _ensure_pretty_token_column(vocab_df: pl.DataFrame) -> pl.DataFrame:
        """Add ``pretty_token`` column if absent (e.g. after loading from cache)."""
        if "pretty_token" in vocab_df.columns:
            return vocab_df
        return vocab_df.with_columns(
            pl.when(pl.col("source_name") == "__special__")
            .then(pl.col("token"))
            .otherwise(
                pl.col("token").str.split_exact("__", 1).struct.field("field_1")
            )
            .alias("pretty_token")
        )