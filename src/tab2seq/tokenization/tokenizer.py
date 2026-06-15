"""Tokenizer for encoding tabular events into token sequences.

Provides two encoding paths sharing identical tokenization logic:

- ``encode()``:       row-by-row, single entity → ``list[int]`` with CLS/SEP framing.
                      Use for inference, inspection, and collation.
- ``encode_frame()``: vectorized Polars/numpy, whole DataFrame → token_ids list column.
                      Use for batch dataset materialization via ``EventDataset``.

Typical pipeline::

    vocab = Vocabulary(VocabularyConfig(max_vocab_size=50_000, min_token_count=5))
    vocab.fit_from_cohort_train(cohort)

    tok = Tokenizer(vocabulary=vocab)

    # --- Inference (single entity) ---
    ids    = tok.encode(events_df, source_name="hospital")
    padded = tok.pad_sequence(ids, max_length=512)
    tokens = tok.decode(padded)

    # --- Dataset build (whole split, via EventDataset) ---
    source_df = tok.encode_frame(source_df, source_name="hospital", include_token_str=True)
    # source_df now has "token_ids" (List[Int64]) and "token_str" (Utf8) columns
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import TokenizerConfig
from .vocabulary import Vocabulary


class Tokenizer:
    """Encode tabular event rows into integer token sequences.

    The tokenizer is a pure consumer of a fitted :class:`Vocabulary`; it owns
    no token set of its own and has no ``fit`` method.  All vocabulary decisions
    (token set, bin edges, train-split safety) live in :class:`Vocabulary`.

    Two encoding modes:

    ``encode()``
        Row-by-row single-entity path.  Returns a ``list[int]`` framed with
        CLS/SEP.  Suitable for inference and DataLoader collation.

    ``encode_frame()``
        Vectorized whole-DataFrame path.  Adds ``token_ids`` (and optionally
        ``token_str``) columns to the input frame using Polars expressions and
        ``np.searchsorted``.  Suitable for bulk dataset materialization.

    Both paths use identical token naming and bin-assignment logic, so sequences
    produced by ``encode()`` are consistent with those materialized by
    ``encode_frame()``.

    Args:
        vocabulary: Fitted :class:`Vocabulary` instance.
        config: Sequence-construction config (special tokens, excluded columns).
            Defaults to :class:`TokenizerConfig`.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        config: TokenizerConfig | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.config = config or TokenizerConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the fitted vocabulary."""
        return len(self.vocabulary.token2index)

    @property
    def token2index(self) -> dict[str, int]:
        """Delegate to ``vocabulary.token2index``."""
        return self.vocabulary.token2index

    @property
    def index2token(self) -> dict[int, str]:
        """Delegate to ``vocabulary.index2token``."""
        return self.vocabulary.index2token

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(
        self,
        events: pl.DataFrame,
        source_name: str,
        columns: list[str] | None = None,
    ) -> list[int]:
        """Encode event rows for one source into a CLS-framed token ID sequence.

        Output format::

            [CLS, tok_row0_col0, tok_row0_col1, ..., tok_rowN_colM, SEP]

        Null values are silently skipped. Unknown non-null values map to UNK.

        Args:
            events: DataFrame with one or more event rows.
            source_name: Source name matching the vocabulary.
            columns: Columns to encode.  ``None`` → all vocabulary-known feature
                columns for this source (id/excluded columns always skipped).

        Returns:
            List of integer token IDs.
        """
        t2i = self.vocabulary.token2index
        unk_id = self._special_id(self.config.unk_token)
        cls_id = self._special_id(self.config.cls_token)
        sep_id = self._special_id(self.config.sep_token)

        col_categories = self.vocabulary.column_categories(source_name)
        cols = self._resolve_columns(events, columns, col_categories)

        token_ids: list[int] = [cls_id]

        for row in events.iter_rows(named=True):
            for col in cols:
                value = row.get(col)
                if value is None:
                    continue

                if col_categories.get(col) == "continuous_bin":
                    edges = self.vocabulary.bin_edges_for(source_name, col)
                    if edges is None:
                        token_ids.append(unk_id)
                        continue
                    bin_idx = self._assign_bin(float(value), edges)
                    token = f"{source_name}__{col}__BIN_{bin_idx}"
                else:
                    token = f"{source_name}__{col}__{value}"

                token_ids.append(t2i.get(token, unk_id))

        token_ids.append(sep_id)
        return token_ids

    def encode_frame(
        self,
        events: pl.DataFrame,
        source_name: str,
        columns: list[str] | None = None,
        include_token_str: bool = False,
    ) -> pl.DataFrame:
        """Vectorized encoding: adds ``token_ids`` (and optionally ``token_str``) columns.

        Returns the input DataFrame with new columns appended.  All intermediate
        working columns are dropped before returning.  Column selection, token
        naming, and bin-assignment logic are identical to :meth:`encode`.

        Categorical columns are encoded via a Polars string-concatenation
        expression.  Continuous columns use ``np.searchsorted`` over the full
        column at once, then a pre-built ``pl.Series`` for the token strings.
        Token strings are mapped to IDs via ``replace_strict``.

        Args:
            events: DataFrame for one source (typically one split worth of rows).
            source_name: Source name matching the vocabulary.
            columns: Columns to encode.  ``None`` → auto-resolve (same rules as
                :meth:`encode`).
            include_token_str: If ``True``, also add a ``token_str`` column
                containing space-joined human-readable token strings.

        Returns:
            Input DataFrame with ``token_ids: List[Int64]`` added, and
            ``token_str: Utf8`` if ``include_token_str`` is ``True``.
        """
        col_categories = self.vocabulary.column_categories(source_name)
        cols = self._resolve_columns(events, columns, col_categories)

        t2i = self.vocabulary.token2index
        unk_id = self._special_id(self.config.unk_token)

        token_str_cols: list[str] = []
        token_id_cols: list[str] = []
        exprs: list[pl.Expr] = []

        for col in cols:
            cat = col_categories[col]
            prefix = f"{source_name}__{col}__"
            tok_alias = f"__tok_{col}"
            id_alias = f"__tid_{col}"

            if cat == "categorical":
                exprs.append(
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.lit(prefix) + pl.col(col).cast(pl.Utf8))
                    .otherwise(pl.lit(None, dtype=pl.Utf8))
                    .alias(tok_alias)
                )
                token_str_cols.append(tok_alias)
                token_id_cols.append(id_alias)

            elif cat == "continuous_bin":
                edges = self.vocabulary.bin_edges_for(source_name, col)
                if edges is None or edges.size < 2:
                    continue

                vals = events[col].fill_null(0.0).to_numpy()
                bin_indices = np.clip(
                    np.searchsorted(edges, vals, side="right") - 1,
                    0,
                    edges.size - 2,
                )
                tok_series = pl.Series(
                    tok_alias,
                    [f"{prefix}BIN_{i}" for i in bin_indices],
                    dtype=pl.Utf8,
                )
                exprs.append(
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.lit(tok_series))
                    .otherwise(pl.lit(None, dtype=pl.Utf8))
                    .alias(tok_alias)
                )
                token_str_cols.append(tok_alias)
                token_id_cols.append(id_alias)

        if exprs:
            events = events.with_columns(exprs)

        # Map token strings → integer IDs
        if token_id_cols:
            events = events.with_columns([
                pl.when(pl.col(tok_col).is_not_null())
                .then(pl.col(tok_col).replace_strict(t2i, default=unk_id).cast(pl.Int64))
                .otherwise(pl.lit(None, dtype=pl.Int64))
                .alias(id_col)
                for tok_col, id_col in zip(token_str_cols, token_id_cols)
            ])

        # Build token_ids list column (drop per-row nulls)
        if token_id_cols:
            events = events.with_columns(
                pl.concat_list([pl.col(c) for c in token_id_cols])
                .list.eval(pl.element().drop_nulls())
                .alias("token_ids")
            )
        else:
            events = events.with_columns(
                pl.lit(None, dtype=pl.List(pl.Int64)).alias("token_ids")
            )

        # Optionally build token_str column
        if include_token_str:
            if token_str_cols:
                events = events.with_columns(
                    pl.concat_list([pl.col(c) for c in token_str_cols])
                    .list.eval(pl.element().drop_nulls())
                    .list.join(" ")
                    .alias("token_str")
                )
            else:
                events = events.with_columns(
                    pl.lit("", dtype=pl.Utf8).alias("token_str")
                )

        # Drop all intermediate working columns
        intermediates = [c for c in token_str_cols + token_id_cols if c in events.columns]
        if intermediates:
            events = events.drop(intermediates)

        return events

    def decode(self, token_ids: list[int]) -> list[str]:
        """Decode integer token IDs back to token strings.

        Unknown IDs return the UNK token string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            List of token strings of the same length.
        """
        i2t = self.vocabulary.index2token
        unk = self.config.unk_token
        return [i2t.get(tid, unk) for tid in token_ids]

    def pad_sequence(self, token_ids: list[int], max_length: int) -> list[int]:
        """Pad or truncate a sequence to exactly ``max_length``.

        Truncation: preserves CLS at index 0, overwrites last slot with SEP.
        Padding: appends PAD tokens to the right.

        Args:
            token_ids: Sequence from :meth:`encode`.
            max_length: Target length (must be ≥ 2).

        Returns:
            List of exactly ``max_length`` token IDs.
        """
        sep_id = self._special_id(self.config.sep_token)
        pad_id = self._special_id(self.config.pad_token)

        if len(token_ids) >= max_length:
            return token_ids[:max_length - 1] + [sep_id]
        return token_ids + [pad_id] * (max_length - len(token_ids))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _special_id(self, token: str) -> int:
        t2i = self.vocabulary.token2index
        if token not in t2i:
            raise KeyError(
                f"Special token '{token}' missing from vocabulary. "
                "Ensure VocabularyConfig.special_tokens includes it before fitting."
            )
        return t2i[token]

    @staticmethod
    def _assign_bin(value: float, edges: np.ndarray) -> int:
        """Map a float to a bin index, identical to fit-time logic.

        Uses ``searchsorted(side='right') - 1`` clipped to ``[0, n_bins - 1]``,
        matching :meth:`Vocabulary._collect_continuous_tokens`.
        """
        idx = int(np.searchsorted(edges, value, side="right")) - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    def _resolve_columns(
        self,
        events: pl.DataFrame,
        columns: list[str] | None,
        col_categories: dict[str, str],
    ) -> list[str]:
        """Return the ordered list of columns to encode.

        When ``columns`` is ``None``, only columns present in both the DataFrame
        and the vocabulary for this source are included; id and excluded columns
        are always filtered out.
        """
        excluded = set(self.config.exclude_columns) | set(self.config.id_columns)

        if columns is not None:
            return [
                c for c in columns
                if c in events.columns and c not in excluded and c in col_categories
            ]

        return [
            col for col in events.columns
            if col not in excluded and col in col_categories
        ]