"""Project-level configuration models for vocabulary and tokenization."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VocabularyConfig(BaseModel):
    """Vocabulary-building configuration.

    Attributes:
        max_vocab_size: Hard cap on total tokens (includes special tokens).
        min_token_count: Minimum train-split occurrences for a token to be retained.
        special_tokens: Reserved tokens, assigned IDs 0..N in order.
    """

    max_vocab_size: int = Field(default=50_000, gt=10)
    min_token_count: int = Field(default=1, ge=1)
    special_tokens: list[str] = Field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )


class TokenizerConfig(BaseModel):
    """Tokenizer configuration.

    Note:
        Vocabulary size is controlled by :class:`VocabularyConfig`; this config
        governs only sequence-construction behaviour (framing, column filtering).

    Attributes:
        id_columns: Columns treated as entity identifiers — always excluded from
            tokenization regardless of vocabulary content.
        exclude_columns: Additional columns to skip during encoding.
        pad_token: Token string for padding.
        unk_token: Token string for unknown values.
        cls_token: Sequence-start token.
        sep_token: Sequence-end token.
        mask_token: Mask token for MLM pre-training.
        vocabulary: Embedded vocabulary configuration (used when building a
            :class:`~tab2seq.tokenizer.vocabulary.Vocabulary` inline).
    """

    id_columns: list[str] = Field(default_factory=lambda: ["entity_id"])
    exclude_columns: list[str] = Field(default_factory=list)
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)