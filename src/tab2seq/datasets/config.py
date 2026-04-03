"""Configuration models for event dataset building."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RelativeDateRule(BaseModel):
    """Specify the rule to compute a relative time feature from event timestamp and a static date column.
    For example, how to compute patient age at event time from event timestamp and patient birthdate.
    Args:
        source_static_column: Name of the static column containing the reference date (e.g. birthdate).
        output_column: Name of the output column to store the computed relative date.
        unit: Unit of time for the relative date calculation ('days', 'weeks', 'months', 'years').
        floor_int: If True, floor the result to the nearest integer.
    """

    source_static_column: str = Field(min_length=1)
    output_column: str = Field(min_length=1)
    unit: Literal["days", "weeks", "months", "years"] = "years"
    floor_int: bool = True


class EventDatasetConfig(BaseModel):
    """Configuration for building and persisting tokenized event datasets.
    Args:
        reference_date: Reference date used for computing time features (for the primary timestamp). Must be in ISO format (YYYY-MM-DD).
        threshold_date: Threshold date used to filter out events occurring after this date. Must be in ISO format (YYYY-MM-DD).
        include_after_threshold: If True, include events occurring after the threshold date instead of filtering them out.
        include_token_str: Whether to include human-readable token strings in the output dataset for interpretability.
        embed_static_in_events: Whether to include static attributes in each event row (denormalized) or keep them in a separate static table.
        relative_date_features: List of rules to compute relative date features from event timestamps and static date columns.
        output_dir: Optional directory path to persist the built dataset artifacts. If not provided, artifacts will not be persisted.
    """

    reference_date: str = "1970-01-01"
    threshold_date: str = "2100-01-01"
    include_after_threshold: bool = True
    include_token_str: bool = True
    embed_static_in_events: bool = False
    relative_date_features: list[RelativeDateRule] = Field(default_factory=list)
    output_dir: Path | None = None

    @field_validator("reference_date", "threshold_date")
    @classmethod
    def _validate_iso_date(cls, v: str) -> str:
        date.fromisoformat(v)
        return v

    def config_hash(self) -> str:
        """Deterministic hash used for dataset cache folder naming."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
