"""Core Source class for data access and validation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from tab2seq.source.config import SourceConfig

logger = logging.getLogger(__name__)


class SchemaError(Exception):
    """Raised when source data does not match the expected schema."""


class Source:
    """A single data source with schema validation and chunked access.

    Uses Polars internally for lazy evaluation and efficient I/O.

    Parameters:
        config: Configuration describing this source.

    Note:
        Rows with null entity IDs or timestamps are automatically
        dropped during scanning and reading. Entity IDs are cast
        to string regardless of the original dtype.

    Example::

        source = Source(SourceConfig(
            name="health",
            filepath=Path("data/health.parquet"),
            entity_id_col="pid",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis", "department"],
            continuous_cols=[],
            file_format="parquet",
            dtype_overrides={"diagnosis": "str"},
        ))

        # lazy scan
        lf = source.scan()

        # eager read
        df = source.read_all()

        # chunked iteration
        for chunk in source.iter_chunks(chunk_size=50_000):
            process(chunk)
    """

    def __init__(self, config: SourceConfig) -> None:
        if not isinstance(config, SourceConfig):
            raise TypeError(f"Expected SourceConfig, got {type(config).__name__}")
        self.config = config

    def __repr__(self) -> str:
        return f"Source(name={self.config.name!r}, path={self.config.filepath!r})"

    @property
    def name(self) -> str:
        """Shortcut to ``self.config.name``."""
        return self.config.name

    @property
    def columns(self) -> list[str]:
        """All columns to select from the source file."""
        return [
            self.config.entity_id_col,
            *(self.config.timestamp_cols or []),
            *(self.config.categorical_cols or []),
            *(self.config.continuous_cols or []),
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_schema(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        """Check that expected columns are present.

        Args:
            df: A DataFrame or LazyFrame from this source.

        Raises:
            SchemaError: If any required column is missing.
        """
        columns = (
            df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
        )
        available = set(columns)
        expected = set(self.columns)
        missing = expected - available
        if missing:
            msg = f"Source '{self.name}' is missing columns: {missing}"
            raise SchemaError(msg)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def scan(self) -> pl.LazyFrame:
        """Return a lazy scan of the source file.

        This is the most idiomatic Polars entry point. No data is
        read until ``.collect()`` is called.

        Note:
            Rows with null entity IDs are automatically dropped. If a timestamp
            column is defined, rows with null timestamps are also dropped.
            Entity IDs are cast to string regardless of the original dtype.

        Returns:
            LazyFrame selecting only the relevant columns, sorted by entity ID
            and timestamp (if defined).
        """
        lf = self._scan_raw()
        self.validate_schema(lf)

        lf = (
            lf.select(self.columns)
            .cast({self.config.entity_id_col: pl.Utf8})
            .drop_nulls(subset=[
                col for col in [self.config.entity_id_col, *(self.config.timestamp_cols or [])] if col is not None
            ])
        )
        # TODO: double check
        # The list comprehension building the subset for drop_nulls creates a list that 
        # could contain None values, which would then be passed to drop_nulls. 
        # While this might work in Polars, it's cleaner to filter out 
        # None values explicitly rather than relying on implicit handling. 
        # The logic should ensure only valid column names are passed.

        sort_by = [self.config.entity_id_col]
        if self.config.timestamp_cols is not None:
            sort_by.extend(self.config.timestamp_cols)

        return lf.sort(sort_by)

    def read_all(self) -> pl.DataFrame:
        """Eagerly read the entire source into a DataFrame.

        Prefer :meth:`scan` or :meth:`iter_chunks` for large files.

        Returns:
            Full DataFrame for this source.
        """
        return self.scan().collect()

    def iter_chunks(self, chunk_size: int = 100_000) -> Iterator[pl.DataFrame]:
        """Yield DataFrames of approximately ``chunk_size`` rows.

        Args:
            chunk_size: Maximum number of rows per chunk.

        Yields:
            Polars DataFrames of up to ``chunk_size`` rows.
        """
        df = self.read_all()
        n_rows = df.height
        for offset in range(0, n_rows, chunk_size):
            yield df.slice(offset, chunk_size)

    def get_entity_ids(self) -> set[str]:
        """Collect all unique entity IDs from this source.

        Note:
            Rows with null entity IDs or timestamps are automatically
            dropped during scanning and reading. Entity IDs are cast
            to string regardless of the original dtype. 
            
            If a timestamp column is defined, only rows with non-null timestamps are
            considered when collecting entity IDs.

        Returns:
            Set of unique entity identifiers.
        """
        lf = (
            self._scan_raw()
            .select([self.config.entity_id_col, *(
                self.config.timestamp_cols if self.config.timestamp_cols else []
            )])
            .cast({self.config.entity_id_col: pl.Utf8})
        )

        if self.config.timestamp_cols is not None:
            logger.info("Source '%s': scanning for unique entities with timestamp filtering", self.name)
            lf = lf.drop_nulls(subset=self.config.timestamp_cols)

        ids = (
            lf
            .drop_nulls(subset=[self.config.entity_id_col])
            .select(self.config.entity_id_col)
            .unique()
            .collect()
            .get_column(self.config.entity_id_col)
            .to_list()
        )
        logger.info("Source '%s': found %d unique entities", self.name, len(ids))
        return set(ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_raw(self) -> pl.LazyFrame:
        """Return a raw lazy scan without column selection or sorting."""
        filepath = Path(self.config.filepath)
        if filepath.suffix == ".parquet":
            return pl.scan_parquet(filepath)
        if filepath.suffix == ".csv":
            return pl.scan_csv(
                filepath,
                has_header=True,
                # dtypes=self.config.dtype_overrides,
            )
        msg = f"Unsupported file format: {filepath.suffix}"
        raise ValueError(msg)
