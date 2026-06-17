"""Project-level configuration models for vocabulary and tokenization."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class VocabularyConfig(BaseModel):
    """Vocabulary-building configuration.

    Attributes:
        max_vocab_size: Hard cap on total tokens (includes special tokens).
        min_token_count: Minimum train-split occurrences for a token to be retained.
        count_mode: Token counting mode used for ``min_token_count`` filtering.
            ``"overall"`` counts every token occurrence (e.g., multiple occurrences within the same entity).
            ``"entity_unique"`` counts each token at most once per entity.
        pad_token: Padding token string.
        unk_token: Unknown-value token string.
        cls_token: Sequence-start token string.
        sep_token: Sequence-end token string.
        mask_token: Mask token for MLM pre-training.
        extra_tokens: Additional reserved tokens appended after the standard five.
            Use for domain-specific sentinels that must always be in the vocabulary
            regardless of training-data content (e.g. ``"[DEATH]"``, ``"[RETIRED]"``).
            These tokens are assigned IDs immediately after the standard tokens and
            are never filtered by ``min_token_count``.
    """

    max_vocab_size: int = Field(default=50_000, gt=10)
    min_token_count: int = Field(default=1, ge=1)
    count_mode: Literal["overall", "entity_unique"] = "overall"
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    extra_tokens: list[str] = Field(default_factory=list)

    @property
    def special_tokens(self) -> list[str]:
        """Ordered list of all reserved tokens: standard five followed by extra_tokens."""
        return [
            self.pad_token, self.unk_token, self.cls_token,
            self.sep_token, self.mask_token,
            *self.extra_tokens,
        ]

    @model_validator(mode="after")
    def _no_duplicate_special_tokens(self) -> "VocabularyConfig":
        tokens = self.special_tokens
        seen: set[str] = set()
        dupes: set[str] = set()
        for t in tokens:
            (dupes if t in seen else seen).add(t)
        if dupes:
            raise ValueError(
                f"special_tokens must be unique; duplicates found: {sorted(dupes)}"
            )
        return self


class TokenizerConfig(BaseModel):
    """Tokenizer configuration.

    Note:
        Vocabulary size and special-token strings are controlled by
        :class:`VocabularyConfig`; this config governs only column filtering.

    Attributes:
        id_columns: Columns treated as entity identifiers — always excluded from
            tokenization regardless of vocabulary content.
        exclude_columns: Additional columns to skip during encoding.
        vocabulary: Embedded vocabulary configuration (used when building a
            :class:`~tab2seq.tokenizer.vocabulary.Vocabulary` inline).
    """

    id_columns: list[str] = Field(default_factory=lambda: ["entity_id"])
    exclude_columns: list[str] = Field(default_factory=list)
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)