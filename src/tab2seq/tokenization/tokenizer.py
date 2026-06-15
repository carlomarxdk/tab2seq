"""Tokenizer module for Life2Vec-style token sequences."""
from __future__ import annotations

from typing import Optional

import polars as pl

from .config import TokenizerConfig


class Tokenizer:
    """Tokenize tabular events into Life2Vec-style token sequences."""

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """Initialize tokenizer.

        Args:
            config: Tokenizer configuration. If None, uses default configuration.
        """
        self.config = config or TokenizerConfig()
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self._build_special_tokens()

    def _build_special_tokens(self) -> None:
        """Build special tokens vocabulary."""
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.cls_token,
            self.config.sep_token,
            self.config.mask_token,
        ]

        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token

    def _resolve_columns(
        self,
        available_columns: list[str],
        columns: Optional[list[str]] = None,
    ) -> list[str]:
        """Resolve tokenization columns and skip identifier columns by default."""
        if columns is None:
            id_columns = set(self.config.id_columns)
            return [col for col in available_columns if col not in id_columns]
        return [col for col in columns if col in available_columns]

    def fit(
        self, events: list[pl.DataFrame], columns: Optional[list[str]] = None
    ) -> None:
        """Build vocabulary from event data.

        Args:
            events: List of DataFrames with events
            columns: Columns to use for vocabulary. If None, uses all columns.
        """
        seen_tokens: set[str] = set(self.vocab.keys())
        next_id = len(self.vocab)

        for df in events:
            cols = self._resolve_columns(df.columns, columns)

            for col in cols:
                for value in df[col].unique():
                    token = f"{col}_{value}"
                    if token not in seen_tokens:
                        if next_id < self.config.vocab_size:
                            self.vocab[token] = next_id
                            self.reverse_vocab[next_id] = token
                            seen_tokens.add(token)
                            next_id += 1
                        else:
                            break

                if next_id >= self.config.vocab_size:
                    break

            if next_id >= self.config.vocab_size:
                break

    def encode(
        self, events: pl.DataFrame, columns: Optional[list[str]] = None
    ) -> list[int]:
        """Encode events into token IDs.

        Args:
            events: DataFrame with events for a single person
            columns: Columns to use for tokenization. If None, uses all columns.

        Returns:
            List of token IDs
        """
        cols = self._resolve_columns(events.columns, columns)

        tokens = [self.vocab[self.config.cls_token]]

        for row in events.iter_rows(named=True):
            for col in cols:
                token = f"{col}_{row[col]}"
                token_id = self.vocab.get(token, self.vocab[self.config.unk_token])
                tokens.append(token_id)

        tokens.append(self.vocab[self.config.sep_token])
        return tokens

    def decode(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs back to tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            List of token strings
        """
        return [self.reverse_vocab.get(tid, self.config.unk_token) for tid in token_ids]

    def pad_sequence(self, token_ids: list[int], max_length: int) -> list[int]:
        """Pad or truncate sequence to max_length.

        Args:
            token_ids: List of token IDs
            max_length: Maximum sequence length

        Returns:
            Padded/truncated sequence
        """
        if len(token_ids) > max_length:
            # Keep CLS token and truncate, then add SEP
            return token_ids[: max_length - 1] + [self.vocab[self.config.sep_token]]
        else:
            # Pad with PAD tokens
            padding = [self.vocab[self.config.pad_token]] * (
                max_length - len(token_ids)
            )
            return token_ids + padding

    @property
    def vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
