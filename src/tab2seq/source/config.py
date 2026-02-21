from __future__ import annotations

from typing import Any
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class SourceConfig(BaseModel):
    """Schema description for a single data source.

    Attributes:
        name (str): Human-readable identifier for this source.
        filepath (Path | str): Path to the data file
        entity_id_col (str): Column name for entity IDs (e.g., person_id, patient_id)
        timestamp_cols (list[str]): Column names for timestamps
        categorical_cols (list[str]): Column names for categorical event features (e.g, ["diagnosis", "procedure", "department"])
        continuous_cols (list[str]): Column names for continuous event features (e.g., ["cost", "length_of_stay"])
        output_format (str): Storage format, either ``"parquet"`` or ``"csv"``.
        dtype_overrides (dict[str, str] | None): Optional dtype mapping passed to the reader.
        preprocessing (dict[str, Any] | None): Optional dict of preprocessing parameters
            interpreted downstream (e.g. ICD truncation level). 
            TODO: Not implemented yet
    """

    name: str
    filepath: Path | str
    entity_id_col: str
    timestamp_cols: list[str] | None = None
    categorical_cols: list[str] | None = None
    continuous_cols: list[str] | None = None
    output_format: str = "parquet"
    dtype_overrides: dict[str, str] | None = None
    preprocessing: dict[str, Any] | None = None

    @field_validator("entity_id_col", "name", "output_format", mode="before")
    @classmethod
    def _validate_non_empty_string(cls, v: str, info: Any) -> str:
        
        field_name = info.field_name
        if not isinstance(v, str):
            msg = f"{field_name} must be a string, got {type(v).__name__}"
            raise ValueError(msg)
        if not v:
            msg = f"{field_name} cannot be empty."
            raise ValueError(msg)
        if v.strip() != v:
            msg = f"{field_name} cannot have leading or trailing whitespace."
            raise ValueError(msg)
        if v.strip() == "":
            msg = f"{field_name} cannot be only whitespace."
            raise ValueError(msg)
        return v

    @field_validator("output_format", mode="before")
    @classmethod
    def _validate_output_format(cls, v: str) -> str:
        allowed = {"parquet", "csv"}
        if v not in allowed:
            msg = f"output_format must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("timestamp_cols", mode="before")
    @classmethod
    def _validate_timestamp_cols(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not isinstance(v, list):
            msg = f"timestamp_cols must be a list of strings, got {type(v).__name__}"
            raise ValueError(msg)
        for col in v:
            if not isinstance(col, str):
                msg = f"Each element in timestamp_cols must be a string, got {type(col).__name__}"
                raise ValueError(msg)
        return v
    
    @field_validator("filepath", mode="before")
    @classmethod
    def _validate_filepath(cls, v: Path | str) -> Path:
        v = Path(v)
        if not v.exists():
            msg = f"File not found: {v}"
            raise FileNotFoundError(msg)
        return v

    @model_validator(mode="after")
    def _check_at_least_one_data_col(self) -> SourceConfig:
        if not self.categorical_cols and not self.continuous_cols:
            msg = "At least one of 'categorical_cols' or 'continuous_cols' must be specified."
            raise ValueError(msg)
        return self

    @property
    def required_columns(self) -> list[str | None]:
        """All columns that must be present in the data."""
        return [
            self.entity_id_col,
        ]
