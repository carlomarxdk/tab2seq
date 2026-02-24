from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class ColumnConfig(BaseModel):
    """Base configuration for a single column.

    Args:
        col_name (str): Name of the column in the source data.
        drop_na (bool): Whether to drop rows with nulls in this column. Defaults to True.
    """

    col_name: str = Field(min_length=1)
    drop_na: bool = False


class CategoricalColConfig(ColumnConfig):
    """Configuration for a single categorical column.
    Args:
        col_name (str): Name of the column in the source data.
        prefix (str): Prefix for the output column name after processing (e.g. "DIAG" → "DIAG_I21.9").
    """

    prefix: str = Field(min_length=1)


class ContinuousColConfig(ColumnConfig):
    """Configuration for a single continuous column.
    Args:
        col_name (str): Name of the column in the source data.
        prefix (str): Prefix for the output column name after processing (e.g. "COST" → "COST_12").
        n_bins (int): Number of bins to use if binning is applied.
        strategy (str): Binning strategy, either "quantile" or "uniform".
    """

    prefix: str = Field(min_length=1)
    n_bins: int = Field(gt=0, default=50)
    strategy: Literal["quantile", "uniform"] = "quantile"


class TimestampColConfig(ColumnConfig):
    is_primary: bool = False

    @model_validator(mode="after")
    def _check_primary_timestamp(self) -> TimestampColConfig:
        if self.is_primary and not self.drop_na:
            raise ValueError(
                "Primary timestamp column must have 'drop_na' set to True."
            )
        return self


class SourceConfig(BaseModel):
    """Schema description for a single data source.

    Attributes:
        name (str): Human-readable identifier for this source.
        filepath (Path | str): Path to the data file
        id_col (str): Column name for entity IDs (e.g., person_id, patient_id)

        timestamp_cols (list[TimestampColConfig]): Column names for timestamps
        categorical_cols (list[CategoricalColConfig]): Column names for categorical event features
        continuous_cols (list[ContinuousColConfig]): Column names for continuous event features
        output_format (str): Storage format, either ``"parquet"`` or ``"csv"``.

    Example::
        config = SourceConfig(
                name="health",
                filepath=Path("data/health.parquet"),
                id_col="entity_id",
                timestamp_cols=[TimestampColConfig(col_name="date", is_primary=True)],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                    CategoricalColConfig(col_name="department", prefix="DEPT"),
                ],
                continuous_cols=[
                    ContinuousColConfig(col_name="cost", prefix="COST", n_bins=20, strategy="quantile")
                ],
                output_format="parquet",
            )
    """

    name: str
    filepath: Path | str
    id_col: str
    categorical_cols: list[CategoricalColConfig] | None = None
    continuous_cols: list[ContinuousColConfig] | None = None
    timestamp_cols: list[TimestampColConfig] | None = None
    output_format: Literal["parquet", "csv"] = "parquet"
    output_folder: Path | str = Path("data/sources/")

    @field_validator("name", "id_col", mode="before")
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

    @field_validator("filepath", mode="before")
    @classmethod
    def _validate_filepath(cls, v: Path | str) -> Path:
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    @field_validator("output_format", mode="before")
    @classmethod
    def _validate_output_format(cls, v: str) -> str:
        allowed = {"parquet", "csv"}
        if v not in allowed:
            msg = f"output_format must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_at_least_one_data_col(self) -> SourceConfig:
        if not self.categorical_cols and not self.continuous_cols:
            msg = "At least one of 'categorical_cols' or 'continuous_cols' must be specified."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_at_most_one_primary_timestamp(self) -> SourceConfig:
        if not self.timestamp_cols:
            return self
        primary = [col for col in self.timestamp_cols if col.is_primary]
        if len(primary) > 1:
            names = [col.col_name for col in primary]
            raise ValueError(
                f"At most one timestamp column can be primary, got {len(primary)}: {names}"
            )
        return self

    @model_validator(mode="after")
    def _check_no_duplicate_col_names(self) -> SourceConfig:
        """Ensure col_name is unique across all column configs and id_col."""
        all_names: list[str] = [self.id_col]
        for group in (self.timestamp_cols, self.categorical_cols, self.continuous_cols):
            if group:
                all_names.extend(col.col_name for col in group)

        seen, duplicates = set(), set()
        for name in all_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)

        if duplicates:
            raise ValueError(
                f"Duplicate column names found across configs: {sorted(duplicates)}"
            )
        return self

    @property
    def cols(self) -> list[str]:
        """All column names required from the source file."""
        names = [self.id_col]
        for group in (self.timestamp_cols, self.categorical_cols, self.continuous_cols):
            if group:
                names.extend(col.col_name for col in group)
        return names

    @property
    def primary_timestamp(self) -> TimestampColConfig | None:
        """The primary timestamp column config, if any."""
        if not self.timestamp_cols:
            return None
        return next((col for col in self.timestamp_cols if col.is_primary), None)
