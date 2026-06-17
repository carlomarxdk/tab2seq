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
        reference_date: Reference date for computing ``primary_time`` (days since this
            date per event). Must be ISO format (YYYY-MM-DD).
        threshold_date: Date used to flag post-threshold events.  Must be ISO format.
            Only relevant when ``include_after_threshold=True``.
        include_after_threshold: If ``True``, add an ``after_threshold`` boolean column
            marking events that occur on or after ``threshold_date``.  These events can
            then be filtered at access time via the ``include_after_threshold`` parameter
            on :meth:`~tab2seq.datasets.EventDataset.get_entity_record`.
        relative_date_features: Rules to compute per-event relative time features from
            a static reference date (e.g. age at event from birthdate).  See
            :class:`RelativeDateRule`.
        output_dir: Optional directory for persisted dataset artifacts.  If ``None``,
            artifacts are written alongside the cohort cache.
    """

    reference_date: str = "1970-01-01"
    threshold_date: str = "2100-01-01"
    include_after_threshold: bool = True
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
