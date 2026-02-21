"""Tests for datasets module (synthetic data generation)."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from tab2seq.datasets import generate_synthetic_data, generate_synthetic_collections
from tab2seq.source import SourceCollection


class TestGenerateSyntheticCollections:
    """Tests for generate_synthetic_collections function."""

    def test_generate_synthetic_collections_basic(self):
        """Test basic synthetic data generation returning SourceCollection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=10, seed=42
            )

            assert isinstance(collection, SourceCollection)
            assert len(collection) == 4  # health, income, labour, survey
            assert "health" in collection
            assert "income" in collection
            assert "labour" in collection
            assert "survey" in collection

    def test_generate_synthetic_collections_files_created(self):
        """Test that parquet files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            generate_synthetic_collections(output_dir=tmpdir, n_entities=10, seed=42)

            assert (tmpdir / "health.parquet").exists()
            assert (tmpdir / "income.parquet").exists()
            assert (tmpdir / "labour.parquet").exists()
            assert (tmpdir / "survey.parquet").exists()

    def test_generate_synthetic_collections_n_entities(self):
        """Test that the correct number of entities are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_entities = 50
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=n_entities, seed=42
            )

            # Check that we have entity IDs
            all_ids = collection.get_all_entity_ids()
            # Note: some entities might not have events in all registries
            # But we should have generated IDs for n_entities
            assert len(all_ids) <= n_entities
            assert len(all_ids) > 0

    def test_generate_synthetic_collections_health_structure(self):
        """Test health registry data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=20, seed=42
            )

            health = collection["health"]
            df = health.read_all()

            # Check required columns
            assert "entity_id" in df.columns
            assert "date" in df.columns
            assert "diagnosis" in df.columns
            assert "procedure" in df.columns
            assert "department" in df.columns
            assert "cost" in df.columns
            assert "length_of_stay" in df.columns

            # Check data types and values
            assert df.height > 0
            assert df["cost"].dtype in [pl.Float64, pl.Float32]

    def test_generate_synthetic_collections_income_structure(self):
        """Test income registry data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=20, seed=42
            )

            income = collection["income"]
            df = income.read_all()

            # Check required columns
            assert "entity_id" in df.columns
            assert "year" in df.columns
            assert "income_type" in df.columns
            assert "sector" in df.columns
            assert "income_amount" in df.columns

            # Check that income amounts are reasonable
            assert df["income_amount"].min() >= 0

    def test_generate_synthetic_collections_labour_structure(self):
        """Test labour registry data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=20, seed=42
            )

            labour = collection["labour"]
            df = labour.read_all()

            # Check required columns
            assert "entity_id" in df.columns
            assert "date" in df.columns
            assert "status" in df.columns
            assert "occupation" in df.columns
            assert "weekly_hours" in df.columns
            assert "residence_region" in df.columns

            # Check data integrity
            assert df["weekly_hours"].min() >= 0

    def test_generate_synthetic_collections_survey_structure(self):
        """Test survey data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=20, seed=42
            )

            survey = collection["survey"]
            df = survey.read_all()

            # Check required columns
            assert "entity_id" in df.columns
            assert "survey_date" in df.columns
            assert "education_level" in df.columns
            assert "marital_status" in df.columns
            assert "self_rated_health" in df.columns
            assert "satisfaction_score" in df.columns

    def test_generate_synthetic_collections_subset_registries(self):
        """Test generating only a subset of registries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir,
                n_entities=10,
                seed=42,
                registries=["health", "income"],
            )

            assert len(collection) == 2
            assert "health" in collection
            assert "income" in collection
            assert "labour" not in collection
            assert "survey" not in collection

    def test_generate_synthetic_collections_invalid_registry(self):
        """Test that invalid registry name raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown registries"):
                generate_synthetic_collections(
                    output_dir=tmpdir,
                    n_entities=10,
                    registries=["invalid_registry"],
                )

    def test_generate_synthetic_collections_reproducibility(self):
        """Test that same seed produces same data."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                seed = 12345
                collection1 = generate_synthetic_collections(
                    output_dir=tmpdir1, n_entities=20, seed=seed
                )
                collection2 = generate_synthetic_collections(
                    output_dir=tmpdir2, n_entities=20, seed=seed
                )

                # Compare health data
                df1 = collection1["health"].read_all()
                df2 = collection2["health"].read_all()

                assert df1.height == df2.height
                # Compare a few rows to ensure reproducibility
                assert df1.head().equals(df2.head())

    def test_generate_synthetic_collections_entity_id_format(self):
        """Test entity ID format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=100, seed=42
            )

            health = collection["health"]
            df = health.read_all()
            entity_ids = df["entity_id"].unique().to_list()

            # Check format: E001, E002, ..., E100
            for eid in entity_ids:
                assert eid.startswith("E")
                assert eid[1:].isdigit()

    def test_generate_synthetic_collections_temporal_ordering(self):
        """Test that events are temporally ordered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=20, seed=42
            )

            health = collection["health"]
            df = health.read_all()

            # Check that data is sorted by entity_id and date
            # Group by entity and check dates are sorted
            for entity_id_tuple, group in df.group_by("entity_id"):
                entity_id = entity_id_tuple[0]
                dates = group["date"].to_list()
                assert dates == sorted(dates), f"Dates not sorted for {entity_id}"

    def test_generate_synthetic_collections_config_compatibility(self):
        """Test that generated sources have proper configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=10, seed=42
            )

            # Check that each source has valid configuration
            for source in collection:
                assert source.config.name in [
                    "health",
                    "income",
                    "labour",
                    "survey",
                ]
                assert source.config.entity_id_col == "entity_id"
                assert len(source.config.categorical_cols) > 0

    def test_generate_synthetic_collections_realistic_values(self):
        """Test that generated values are realistic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=50, seed=42
            )

            # Check health costs are positive and realistic
            health = collection["health"]
            df = health.read_all()
            costs = df["cost"].to_list()
            assert all(c > 0 for c in costs)
            assert max(costs) > 100  # Should have some significant costs

            # Check income amounts are realistic
            income = collection["income"]
            df = income.read_all()
            amounts = df["income_amount"].to_list()
            assert all(a >= 0 for a in amounts)

    def test_generate_synthetic_collections_survey_participation(self):
        """Test survey participation rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_entities = 100
            collection = generate_synthetic_collections(
                output_dir=tmpdir, n_entities=n_entities, seed=42
            )

            survey = collection["survey"]
            df = survey.read_all()
            unique_participants = df["entity_id"].n_unique()

            # Survey should have ~60% participation rate per wave
            # With 5 waves (2016, 2018, 2020, 2022, 2024), expect significant participation
            assert unique_participants > 0
            assert unique_participants <= n_entities


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data function."""

    def test_generate_synthetic_data_basic(self):
        """Test that generate_synthetic_data returns dict of paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_synthetic_data(
                output_dir=tmpdir, n_entities=10, seed=42
            )

            assert isinstance(paths, dict)
            assert set(paths.keys()) == {"health", "income", "labour", "survey"}

    def test_generate_synthetic_data_files_exist(self):
        """Test that generated files exist at returned paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_synthetic_data(
                output_dir=tmpdir, n_entities=10, seed=42
            )

            for registry, path in paths.items():
                assert Path(path).exists()
                assert path.suffix == ".parquet"

    def test_generate_synthetic_data_csv_format(self):
        """Test generating CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_synthetic_data(
                output_dir=tmpdir, n_entities=10, seed=42, file_format="csv"
            )

            for registry, path in paths.items():
                assert Path(path).exists()
                assert path.suffix == ".csv"

    def test_generate_synthetic_data_subset_registries(self):
        """Test generating only a subset of registries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_synthetic_data(
                output_dir=tmpdir,
                n_entities=10,
                seed=42,
                registries=["health", "income"],
            )

            assert len(paths) == 2
            assert "health" in paths
            assert "income" in paths
            assert "labour" not in paths
            assert "survey" not in paths

    def test_generate_synthetic_data_invalid_format(self):
        """Test that invalid file format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="file_format must be one of"):
                generate_synthetic_data(
                    output_dir=tmpdir,
                    n_entities=10,
                    file_format="json",
                )

    def test_generate_synthetic_data_reproducibility(self):
        """Test that same seed produces same data."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                seed = 12345
                paths1 = generate_synthetic_data(
                    output_dir=tmpdir1, n_entities=20, seed=seed
                )
                paths2 = generate_synthetic_data(
                    output_dir=tmpdir2, n_entities=20, seed=seed
                )

                # Compare health data
                df1 = pl.read_parquet(paths1["health"])
                df2 = pl.read_parquet(paths2["health"])

                assert df1.height == df2.height
                # Compare a few rows to ensure reproducibility
                assert df1.head().equals(df2.head())
