"""Tests for source module (Source, SourceConfig, SourceCollection)."""

import tempfile
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tab2seq.source import (
    CategoricalColConfig,
    ContinuousColConfig,
    Source,
    SourceCollection,
    SourceConfig,
    TimestampColConfig,
)


@pytest.fixture
def sample_health_data():
    """Create sample health registry data."""
    return pl.DataFrame(
        {
            "patient_id": ["P001", "P001", "P002", "P002", "P003"],
            "date": [
                "2020-01-01",
                "2020-02-01",
                "2020-01-15",
                "2020-03-01",
                "2020-01-20",
            ],
            "diagnosis": ["I21.0", "I21.9", "J18.1", "E11.9", "M54.5"],
            "department": [
                "cardiology",
                "cardiology",
                "pulmonology",
                "endocrinology",
                "orthopedics",
            ],
            "cost": [1500.0, 2000.0, 800.0, 500.0, 300.0],
        }
    )


@pytest.fixture
def sample_income_data():
    """Create sample income registry data."""
    return pl.DataFrame(
        {
            "person_id": ["P001", "P001", "P002", "P003"],
            "year": ["2020-12-31", "2021-12-31", "2020-12-31", "2020-12-31"],
            "income_type": ["salary", "salary", "pension", "self_employment"],
            "amount": [50000.0, 52000.0, 30000.0, 45000.0],
        }
    )


@pytest.fixture
def health_parquet_file(sample_health_data):
    """Create temporary parquet file with health data."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_health_data.write_parquet(f.name)
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def income_csv_file(sample_income_data):
    """Create temporary CSV file with income data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_income_data.write_csv(f.name)
        yield Path(f.name)
    Path(f.name).unlink()


class TestSourceConfig:
    """Tests for SourceConfig."""

    def test_source_config_minimal(self, health_parquet_file):
        """Test minimal SourceConfig."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="department", prefix="DEPT"),
            ],
        )
        assert config.name == "health"
        assert config.id_col == "patient_id"
        assert len(config.timestamp_cols) == 1
        assert config.timestamp_cols[0].col_name == "date"
        assert len(config.categorical_cols) == 2
        assert config.categorical_cols[0].col_name == "diagnosis"
        assert config.categorical_cols[1].col_name == "department"
        assert config.continuous_cols is None or config.continuous_cols == []
        assert config.output_format == "parquet"

    def test_source_config_with_continuous(self, health_parquet_file):
        """Test SourceConfig with continuous columns."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
            continuous_cols=[ContinuousColConfig(col_name="cost", prefix="COST")],
        )
        assert len(config.continuous_cols) == 1
        assert config.continuous_cols[0].col_name == "cost"

    def test_source_config_csv_format(self, income_csv_file):
        """Test SourceConfig with CSV format."""
        config = SourceConfig(
            name="income",
            filepath=income_csv_file,
            id_col="person_id",
            timestamp_cols=[
                TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="income_type", prefix="INCOME")
            ],
            continuous_cols=[ContinuousColConfig(col_name="amount", prefix="AMT")],
            output_format="csv",
        )
        assert config.output_format == "csv"

    def test_source_config_invalid_format(self, health_parquet_file):
        """Test SourceConfig with invalid file format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
                output_format="json",
            )

    def test_source_config_file_not_found(self):
        """Test SourceConfig with non-existent file."""
        with pytest.raises(FileNotFoundError):
            SourceConfig(
                name="health",
                filepath=Path("/nonexistent/file.parquet"),
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
            )

    def test_source_config_no_data_columns(self, health_parquet_file):
        """Test SourceConfig with no data columns."""
        with pytest.raises(ValueError, match="At least one of"):
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=None,
                continuous_cols=None,
            )


class TestSource:
    """Tests for Source class."""

    def test_source_initialization(self, health_parquet_file):
        """Test Source initialization."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="department", prefix="DEPT"),
            ],
            continuous_cols=[ContinuousColConfig(col_name="cost", prefix="COST")],
        )
        source = Source(config)
        assert source.name == "health"
        assert source.config == config

    def test_source_columns_property(self, health_parquet_file):
        """Test Source.columns property."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="department", prefix="DEPT"),
            ],
            continuous_cols=[ContinuousColConfig(col_name="cost", prefix="COST")],
        )
        source = Source(config)
        expected_cols = ["patient_id", "date", "diagnosis", "department", "cost"]
        assert source.columns == expected_cols

    def test_source_scan(self, health_parquet_file):
        """Test Source.scan() returns LazyFrame."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="department", prefix="DEPT"),
            ],
            continuous_cols=[ContinuousColConfig(col_name="cost", prefix="COST")],
        )
        source = Source(config)
        lf = source.scan()
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df.height == 5  # All rows from sample data

    def test_source_read_all(self, health_parquet_file):
        """Test Source.read_all()."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        df = source.read_all()
        assert isinstance(df, pl.DataFrame)
        assert df.height == 5

    def test_source_get_entity_ids(self, health_parquet_file):
        """Test Source.get_entity_ids()."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        entity_ids = source.get_entity_ids()
        assert isinstance(entity_ids, set)
        assert entity_ids == {"P001", "P002", "P003"}

    def test_source_iter_chunks(self, health_parquet_file):
        """Test Source.iter_chunks()."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        chunks = list(source.iter_chunks(chunk_size=2))
        assert len(chunks) == 3  # 5 rows / 2 per chunk = 3 chunks
        assert all(isinstance(chunk, pl.DataFrame) for chunk in chunks)

    def test_source_validate_schema_valid(self, health_parquet_file):
        """Test Source.validate_schema() with valid data."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        lf = source.scan()
        source.validate_schema(lf)  # Should not raise

    def test_source_csv_file(self, income_csv_file):
        """Test Source with CSV file."""
        config = SourceConfig(
            name="income",
            filepath=income_csv_file,
            id_col="person_id",
            timestamp_cols=[
                TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="income_type", prefix="INCOME")
            ],
            continuous_cols=[ContinuousColConfig(col_name="amount", prefix="AMT")],
            output_format="csv",
        )
        source = Source(config)
        df = source.read_all()
        assert df.height == 4
        assert "person_id" in df.columns

    def test_source_process_no_cache(self, health_parquet_file):
        """Test Source.process() without caching."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        df = source.process(cache=False)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 5
        # Verify categorical columns are cast to string
        assert df["diagnosis"].dtype == pl.Utf8

    def test_source_process_prefix_addition(self, health_parquet_file):
        """Test Source.process() adds prefixes to categorical values."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="department", prefix="DEPT"),
            ],
        )
        source = Source(config)
        df = source.process(cache=False)

        # Assert that diagnosis values have DIAG_ prefix
        diagnosis_values = df["diagnosis"].to_list()
        expected_diagnosis = ["DIAG_I21.0", "DIAG_I21.9", "DIAG_J18.1", "DIAG_E11.9", "DIAG_M54.5"]
        assert diagnosis_values == expected_diagnosis

        # Assert that department values have DEPT_ prefix
        department_values = df["department"].to_list()
        expected_department = ["DEPT_cardiology", "DEPT_cardiology", "DEPT_pulmonology", "DEPT_endocrinology", "DEPT_orthopedics"]
        assert department_values == expected_department

    def test_source_process_with_cache_parquet(self, health_parquet_file):
        """Test Source.process() with parquet caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
                output_format="parquet",
                output_folder=tmpdir,
            )
            source = Source(config)

            # First call: should save to cache
            df1 = source.process(cache=True)
            assert isinstance(df1, pl.DataFrame)
            assert df1.height == 5

            # Verify cache file was created
            cache_dir = Path(tmpdir) / "intermediate" / "health"
            cache_files = list(cache_dir.glob("*.parquet"))
            assert len(cache_files) == 1
            mtime_after_first = cache_files[0].stat().st_mtime

            # Second call: should read from cache (file must not be modified)
            df2 = source.process(cache=True)
            assert isinstance(df2, pl.DataFrame)
            assert df2.height == 5
            assert cache_files[0].stat().st_mtime == mtime_after_first

            # Verify the data is the same
            assert_frame_equal(df1, df2)

    def test_source_process_with_cache_csv(self, income_csv_file):
        """Test Source.process() with CSV caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SourceConfig(
                name="income",
                filepath=income_csv_file,
                id_col="person_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="income_type", prefix="INCOME")
                ],
                output_format="csv",
                output_folder=tmpdir,
            )
            source = Source(config)

            # First call: should save to cache
            df1 = source.process(cache=True)
            assert isinstance(df1, pl.DataFrame)
            assert df1.height == 4

            # Verify cache file was created
            cache_dir = Path(tmpdir) / "intermediate" / "income"
            cache_files = list(cache_dir.glob("*.csv"))
            assert len(cache_files) == 1

            # Second call: should read from cache
            df2 = source.process(cache=True)
            assert isinstance(df2, pl.DataFrame)
            assert df2.height == 4


class TestSourceCollection:
    """Tests for SourceCollection class."""

    def test_collection_initialization(self, health_parquet_file):
        """Test SourceCollection initialization."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        source = Source(config)
        collection = SourceCollection([source])
        assert len(collection) == 1

    def test_collection_from_configs(self, health_parquet_file, income_csv_file):
        """Test SourceCollection.from_configs()."""
        configs = [
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                id_col="person_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="income_type", prefix="INCOME")
                ],
                output_format="csv",
            ),
        ]
        collection = SourceCollection.from_configs(configs)
        assert len(collection) == 2
        assert "health" in collection
        assert "income" in collection

    def test_collection_getitem(self, health_parquet_file):
        """Test SourceCollection dict-like access."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        collection = SourceCollection.from_configs([config])
        health_source = collection["health"]
        assert health_source.name == "health"

    def test_collection_getitem_not_found(self, health_parquet_file):
        """Test SourceCollection with non-existent source."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        collection = SourceCollection.from_configs([config])
        with pytest.raises(KeyError, match="Source 'income' not found"):
            _ = collection["income"]

    def test_collection_iteration(self, health_parquet_file, income_csv_file):
        """Test iterating over SourceCollection."""
        configs = [
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                id_col="person_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="income_type", prefix="INCOME")
                ],
                output_format="csv",
            ),
        ]
        collection = SourceCollection.from_configs(configs)
        names = [source.name for source in collection]
        assert set(names) == {"health", "income"}

    def test_collection_names_property(self, health_parquet_file):
        """Test SourceCollection.names property."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            id_col="patient_id",
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
            ],
        )
        collection = SourceCollection.from_configs([config])
        assert collection.names == ["health"]

    def test_collection_get_all_entity_ids(self, health_parquet_file, income_csv_file):
        """Test SourceCollection.get_all_entity_ids()."""
        configs = [
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                id_col="patient_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG")
                ],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                id_col="person_id",
                timestamp_cols=[
                    TimestampColConfig(col_name="year", is_primary=True, drop_na=True)
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="income_type", prefix="INCOME")
                ],
                output_format="csv",
            ),
        ]
        collection = SourceCollection.from_configs(configs)
        all_ids = collection.get_all_entity_ids()
        # Health has P001, P002, P003; Income has P001, P002, P003
        assert all_ids == {"P001", "P002", "P003"}

    def test_collection_from_yaml(self, health_parquet_file, income_csv_file):
        """Test SourceCollection.from_yaml()."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = f"""
sources:
  - name: health
    filepath: {health_parquet_file}
    id_col: patient_id
    timestamp_cols:
      - col_name: date
        is_primary: true
        drop_na: true
    categorical_cols:
      - col_name: diagnosis
        prefix: DIAG
      - col_name: department
        prefix: DEPT
    continuous_cols:
      - col_name: cost
        prefix: COST
    output_format: parquet
  - name: income
    filepath: {income_csv_file}
    id_col: person_id
    timestamp_cols:
      - col_name: year
        is_primary: true
        drop_na: true
    categorical_cols:
      - col_name: income_type
        prefix: INCOME
    continuous_cols:
      - col_name: amount
        prefix: AMT
    output_format: csv
"""
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            collection = SourceCollection.from_yaml(yaml_path)
            assert len(collection) == 2
            assert "health" in collection
            assert "income" in collection
            # Check that categorical_cols are correctly loaded
            health_cat_cols = collection["health"].config.categorical_cols
            assert len(health_cat_cols) == 2
            assert health_cat_cols[0].col_name == "diagnosis"
            assert health_cat_cols[1].col_name == "department"
        finally:
            yaml_path.unlink()
