"""Project-level configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VocabularyConfig(BaseModel):
    """Vocabulary-building configuration."""

    max_vocab_size: int = Field(default=50000, gt=10)
    min_token_count: int = Field(default=1, ge=1)
    special_tokens: list[str] = Field(
        default_factory=lambda: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    
class TokenizerConfig(BaseModel):
    """Tokenizer configuration."""

    vocab_size: int = Field(default=10000, gt=0)
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)