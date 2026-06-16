"""Shared pydantic field-validator helpers used across config modules."""

from __future__ import annotations

from typing import Any


def validate_no_whitespace_string(v: str, info: Any) -> str:
    """Reject empty strings and strings with leading/trailing whitespace."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(
            f"'{info.field_name}' must be a non-empty, non-whitespace string."
        )
    if v != v.strip():
        raise ValueError(
            f"'{info.field_name}' cannot have leading or trailing whitespace."
        )
    return v
