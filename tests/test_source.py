"""Tests for source module (Source, SourceConfig, SourceCollection)."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from tab2seq.source import Source, SourceCollection, SourceConfig, SchemaError


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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis", "department"],
        )
        assert config.name == "health"
        assert config.entity_id_col == "patient_id"
        assert config.timestamp_cols == ["date"]
        assert config.categorical_cols == ["diagnosis", "department"]
        assert config.continuous_cols is None or config.continuous_cols == []
        assert config.output_format == "parquet"

    def test_source_config_with_continuous(self, health_parquet_file):
        """Test SourceConfig with continuous columns."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
            continuous_cols=["cost"],
        )
        assert config.continuous_cols == ["cost"]

    def test_source_config_csv_format(self, income_csv_file):
        """Test SourceConfig with CSV format."""
        config = SourceConfig(
            name="income",
            filepath=income_csv_file,
            entity_id_col="person_id",
            timestamp_cols=["year"],
            categorical_cols=["income_type"],
            continuous_cols=["amount"],
            output_format="csv",
        )
        assert config.output_format == "csv"

    def test_source_config_invalid_format(self, health_parquet_file):
        """Test SourceConfig with invalid file format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=["diagnosis"],
                output_format="json",
            )

    def test_source_config_file_not_found(self):
        """Test SourceConfig with non-existent file."""
        with pytest.raises(FileNotFoundError):
            SourceConfig(
                name="health",
                filepath=Path("/nonexistent/file.parquet"),
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=["diagnosis"],
            )

    def test_source_config_no_data_columns(self, health_parquet_file):
        """Test SourceConfig with no data columns."""
        with pytest.raises(ValueError, match="At least one of"):
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=[],
                continuous_cols=[],
            )


class TestSource:
    """Tests for Source class."""

    def test_source_initialization(self, health_parquet_file):
        """Test Source initialization."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis", "department"],
            continuous_cols=["cost"],
        )
        source = Source(config)
        assert source.name == "health"
        assert source.config == config

    def test_source_columns_property(self, health_parquet_file):
        """Test Source.columns property."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis", "department"],
            continuous_cols=["cost"],
        )
        source = Source(config)
        expected_cols = ["patient_id", "date", "diagnosis", "department", "cost"]
        assert source.columns == expected_cols

    def test_source_scan(self, health_parquet_file):
        """Test Source.scan() returns LazyFrame."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis", "department"],
            continuous_cols=["cost"],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
            continuous_cols=[],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
        )
        source = Source(config)
        lf = source.scan()
        source.validate_schema(lf)  # Should not raise

    def test_source_csv_file(self, income_csv_file):
        """Test Source with CSV file."""
        config = SourceConfig(
            name="income",
            filepath=income_csv_file,
            entity_id_col="person_id",
            timestamp_cols=["year"],
            categorical_cols=["income_type"],
            continuous_cols=["amount"],
            output_format="csv",
        )
        source = Source(config)
        df = source.read_all()
        assert df.height == 4
        assert "person_id" in df.columns


class TestSourceCollection:
    """Tests for SourceCollection class."""

    def test_collection_initialization(self, health_parquet_file):
        """Test SourceCollection initialization."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
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
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=["diagnosis"],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                entity_id_col="person_id",
                timestamp_cols=["year"],
                categorical_cols=["income_type"],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
        )
        collection = SourceCollection.from_configs([config])
        health_source = collection["health"]
        assert health_source.name == "health"

    def test_collection_getitem_not_found(self, health_parquet_file):
        """Test SourceCollection with non-existent source."""
        config = SourceConfig(
            name="health",
            filepath=health_parquet_file,
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
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
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=["diagnosis"],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                entity_id_col="person_id",
                timestamp_cols=["year"],
                categorical_cols=["income_type"],
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
            entity_id_col="patient_id",
            timestamp_cols=["date"],
            categorical_cols=["diagnosis"],
        )
        collection = SourceCollection.from_configs([config])
        assert collection.names == ["health"]

    def test_collection_get_all_entity_ids(self, health_parquet_file, income_csv_file):
        """Test SourceCollection.get_all_entity_ids()."""
        configs = [
            SourceConfig(
                name="health",
                filepath=health_parquet_file,
                entity_id_col="patient_id",
                timestamp_cols=["date"],
                categorical_cols=["diagnosis"],
            ),
            SourceConfig(
                name="income",
                filepath=income_csv_file,
                entity_id_col="person_id",
                timestamp_cols=["year"],
                categorical_cols=["income_type"],
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
    entity_id_col: patient_id
    timestamp_cols:
      - date
    categorical_cols:
      - diagnosis
      - department
    continuous_cols:
      - cost
    output_format: parquet
  - name: income
    filepath: {income_csv_file}
    entity_id_col: person_id
    timestamp_cols:
      - year
    categorical_cols:
      - income_type
    continuous_cols:
      - amount
    output_format: csv
"""
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            collection = SourceCollection.from_yaml(yaml_path)
            assert len(collection) == 2
            assert "health" in collection
            assert "income" in collection
            assert collection["health"].config.categorical_cols == [
                "diagnosis",
                "department",
            ]
        finally:
            yaml_path.unlink()
