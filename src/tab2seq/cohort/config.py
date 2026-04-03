"""Configuration models for cohort definition."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator, field_validator

logger = logging.getLogger("CohortConfig")


class EntityInclusionCriteria(BaseModel):
    """Entity inclusion criteria for a single `Source`.

    Defines requirements an entity must meet within a specific
    `Source` to be included in the cohort.

    Attributes:
        source_name: Name of the `Source` this criteria applies to.
        required: If ``True``, entities must appear in this `Source`.
        min_events: Minimum number of events an entity must have
            in this `Source`. Only checked when ``required`` is ``True``.

    """

    source_name: str
    required: bool = False
    min_events: int | None = None
    max_events: int | None = None

    @field_validator("source_name", mode="before")
    @classmethod
    def _no_whitespace_string(cls, v: str, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"'{info.field_name}' must be a non-empty, non-whitespace string."
            )
        if v != v.strip():
            raise ValueError(
                f"'{info.field_name}' cannot have leading or trailing whitespace."
            )
        return v

    @model_validator(mode="after")
    def _validate_event_bounds(self) -> EntityInclusionCriteria:
        if self.required:
            if self.min_events is None or self.min_events < 1:
                raise ValueError(
                    "If 'required' is True, 'min_events' must be a positive integer."
                )
            if self.max_events is not None and self.max_events < self.min_events:
                raise ValueError(
                    f"'max_events' ({self.max_events}) cannot be less than 'min_events' ({self.min_events})."
                )
        return self


class CohortConfig(BaseModel):
    """Configuration for defining a cohort of entities and splitting into train/val/test sets.
    
    Attributes:
        use_splits: Whether to split the cohort into train/val/test sets.
        train_frac: Fraction of entities to include in the training set.
        val_frac: Fraction of entities to include in the validation set.
        test_frac: Fraction of entities to include in the test set.
        seed: Random seed for reproducible splits.
        stratify_col: Optional column name to use for stratified splitting. 
                      Must be a static column present in the dataset.
    """

    use_splits: bool = True
    train_frac: float = Field(0.7, ge=0.0, le=1.0)
    val_frac: float = Field(0.15, ge=0.0, le=1.0)
    test_frac: float = Field(0.15, ge=0.0, le=1.0)
    seed: int = 792
    stratify_col: str | None = None

    @field_validator("stratify_col", mode="before")
    @classmethod
    def _validate_stratify_col(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("'stratify_col' must be a non-empty string when set.")
        if v != v.strip():
            raise ValueError("'stratify_col' cannot have surrounding whitespace.")
        return v

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> CohortConfig:
        if self.use_splits:
            total = self.train_frac + self.val_frac + self.test_frac
            if abs(total - 1.0) > 1e-12:
                msg = f"Split fractions must sum to 1.0, got {total:.4f}"
                raise ValueError(msg)
        return self

    def config_hash(self) -> str:
        """Deterministic hash of split configuration."""
        payload = json.dumps(self.model_dump(exclude_none=False), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]