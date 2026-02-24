"""Core Source class for data access, validation, and preprocessing."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from tab2seq.source.config import SourceConfig

logger = logging.getLogger(__name__)


class SchemaError(Exception):
    """Raised when source data does not match the expected schema."""


class Source:
    """A single data source with schema validation, chunked access, and preprocessing.

    Uses Polars internally for lazy evaluation and efficient I/O.

    Args:
        config: Configuration describing this source.

    Note:
        Rows with null entity IDs or primary timestamps are automatically
        dropped during scanning and reading. Entity IDs are cast to string
        regardless of the original dtype.

    Example::

        source = Source(
            config=SourceConfig(
                name="health",
                filepath=Path("data/health.parquet"),
                id_col="entity_id",
                timestamp_cols=[
                    TimestampColConfig(
                        col_name="date", is_primary=True, drop_na=True
                    )
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                    CategoricalColConfig(col_name="department", prefix="DEPT"),
                ],
                continuous_cols=[
                    ContinuousColConfig(
                        col_name="cost",
                        prefix="COST",
                        n_bins=20,
                        strategy="quantile",
                    )
                ],
                output_format="parquet",
            )
        )

        # lazy scan
        lf = source.scan()

        # eager read
        df = source.read_all()

        # chunked iteration
        for chunk in source.iter_chunks(chunk_size=50_000):
            process(chunk)

        # preprocess and cache to disk
        df = source.process(cache=True)

    """

    def __init__(self, config: SourceConfig) -> None:
        """Initialize Source with configuration.

        Args:
            config: SourceConfig object describing this source.

        """
        if not isinstance(config, SourceConfig):
            raise TypeError(f"Expected SourceConfig, got {type(config).__name__}")
        self.config = config

    def __repr__(self) -> str:
        """Return string representation of Source."""
        return f"Source(name={self.config.name!r}, path={self.config.filepath!r})"

    @property
    def name(self) -> str:
        """Shortcut to ``self.config.name``."""
        return self.config.name

    @property
    def columns(self) -> list[str]:
        """All columns to select from the source file."""
        return self.config.cols

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
            Rows with null entity IDs are automatically dropped. For timestamp
            columns with ``drop_na=True``, rows with null timestamps are also dropped.
            Entity IDs are cast to string regardless of the original dtype.

        Returns:
            LazyFrame selecting only the relevant columns, sorted by entity ID
            and timestamp (if defined).

        """
        lf = self._scan_raw()
        self.validate_schema(lf)

        ts_col_names = [col.col_name for col in (self.config.timestamp_cols or [])]
        drop_na_cols = [
            col.col_name for col in (self.config.timestamp_cols or []) if col.drop_na
        ]

        lf = (
            lf.select(self.columns)
            .cast({self.config.id_col: pl.Utf8})
            .drop_nulls(subset=[self.config.id_col, *drop_na_cols])
        )

        sort_by = [self.config.id_col, *ts_col_names]
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
            Rows with null entity IDs or timestamps (where ``drop_na=True``) are
            excluded. Entity IDs are cast to string regardless of the original dtype.

        Returns:
            Set of unique entity identifiers.

        """
        ts_col_names = [col.col_name for col in (self.config.timestamp_cols or [])]
        drop_na_cols = [
            col.col_name for col in (self.config.timestamp_cols or []) if col.drop_na
        ]

        lf = (
            self._scan_raw()
            .select([self.config.id_col, *ts_col_names])
            .cast({self.config.id_col: pl.Utf8})
        )

        if drop_na_cols:
            logger.info(
                "Source '%s': scanning for unique entities with timestamp filtering",
                self.name,
            )
            lf = lf.drop_nulls(subset=drop_na_cols)

        ids = (
            lf.drop_nulls(subset=[self.config.id_col])
            .select(self.config.id_col)
            .unique()
            .collect()
            .get_column(self.config.id_col)
            .to_list()
        )
        logger.info("Source '%s': found %d unique entities", self.name, len(ids))
        return set(ids)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def process(self, cache: bool = True) -> pl.DataFrame:
        """Apply preprocessing steps and optionally cache the result.

        Preprocessing steps performed:

        * Rows with nulls in columns configured with ``drop_na=True`` are dropped.
        * Categorical columns are cast to string (``pl.Utf8``) and prefixed with
          ``{prefix}_`` (e.g. ``"I21.9"`` → ``"DIAG_I21.9"``).

        .. note::
            Continuous column binning is intentionally **not** performed here to
            prevent data leakage; it must be applied after the train/test split.

        If *cache* is enabled, the processed DataFrame is saved to a cache
        file in ``{output_folder}/intermediate/{name}/`` (format controlled by
        :attr:`SourceConfig.output_format`). On subsequent calls with the same
        configuration the cached file is read instead of reprocessing.

        Args:
            cache: Whether to cache the processed DataFrame to disk for future reuse.

        Returns:
            Processed :class:`polars.DataFrame`.

        """
        # TODO: Add a test that checks whether the continuous column binning is not applied in this method, and that it is applied in the appropriate place after the train/test split.
        # TODO: find better way to handle cache_path
        cache_path: Path = Path("")
        if cache:
            intermediate_dir = (
                Path(self.config.output_folder) / "intermediate" / self.name
            )
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            ext = self.config.output_format
            cache_path = intermediate_dir / f"{self._config_hash()}.{ext}"

            if cache_path.exists():
                logger.info(
                    "Source '%s': loading preprocessed data from cache %s",
                    self.name,
                    cache_path,
                )
                if ext == "parquet":
                    return pl.read_parquet(cache_path)
                return pl.read_csv(cache_path)

        df = self._apply_preprocessing()

        if cache:
            logger.info(
                "Source '%s': saving preprocessed data to cache %s",
                self.name,
                cache_path,
            )
            if ext == "parquet":
                df.write_parquet(cache_path)
            else:
                df.write_csv(cache_path)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_raw(self) -> pl.LazyFrame:
        """Return a raw lazy scan without column selection or sorting."""
        filepath = Path(self.config.filepath)
        if filepath.suffix == ".parquet":
            return pl.scan_parquet(filepath)
        if filepath.suffix == ".csv":
            return pl.scan_csv(filepath, has_header=True)
        msg = f"Unsupported file format: {filepath.suffix}"
        raise ValueError(msg)

    def _apply_preprocessing(self) -> pl.DataFrame:
        """Read source data and apply all preprocessing transformations."""
        df = self.read_all()

        # Drop nulls for configured columns (categorical and continuous)
        drop_na_cols = [
            col.col_name
            for col in (self.config.categorical_cols or [])
            + (self.config.continuous_cols or [])
            if col.drop_na
        ]
        if drop_na_cols:
            df = df.drop_nulls(subset=drop_na_cols)

        # Cast categorical columns to string and prepend prefix (e.g. "I21.9" -> "DIAG_I21.9")
        for col_cfg in self.config.categorical_cols or []:
            df = df.with_columns(
                (
                    pl.lit(col_cfg.prefix + "_")
                    + pl.col(col_cfg.col_name).cast(pl.Utf8)
                ).alias(col_cfg.col_name)
            )

        return df

    def _config_hash(self) -> str:
        """Return a short hash of the configuration for cache key generation."""
        config_json = self.config.model_dump_json()
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]
