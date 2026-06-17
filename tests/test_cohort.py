"""Tests for cohort module."""

from pathlib import Path

import polars as pl
import pytest

from tab2seq.cohort import Cohort, CohortConfig, EntityInclusionCriteria
from tab2seq.source import (
    CategoricalColConfig,
    ContinuousColConfig,
    SchemaError,
    Source,
    SourceCollection,
    SourceConfig,
    TemporalColConfig,
)


@pytest.fixture
def source_collection(tmp_path: Path) -> SourceCollection:
    """Create a source collection with overlapping entity ids."""
    health_df = pl.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001", "P002", "P002", "P003", "P004"],
            "date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-01-15",
                "2020-02-15",
                "2020-01-20",
                "2020-01-25",
            ],
            "diagnosis": ["I21.0", "I21.9", "I25.1", "J18.1", "E11.9", "M54.5", "C34.1"],
            "birth_date": [
                "1980-05-15",
                "1980-05-15",
                "1980-05-15",
                "1975-03-20",
                "1975-03-20",
                "1990-07-10",
                "1985-11-05",
            ],
        }
    )

    income_df = pl.DataFrame(
        {
            "person_id": ["P001", "P001", "P002", "P003"],
            "year": ["2020-12-31", "2021-12-31", "2020-12-31", "2020-12-31"],
            "income_type": ["salary", "salary", "pension", "self_employment"],
            "amount": [50000.0, 52000.0, 30000.0, 45000.0],
            "region": ["capital", "capital", "north", "south"],
        }
    )

    labour_df = pl.DataFrame(
        {
            "person_id": ["P001", "P002", "P004"],
            "date": ["2020-06-01", "2020-06-01", "2020-06-01"],
            "status": ["employed", "employed", "unemployed"],
            "residence_region": ["capital", "north", "south"],
        }
    )

    health_path = tmp_path / "health.parquet"
    income_path = tmp_path / "income.parquet"
    labour_path = tmp_path / "labour.parquet"

    health_df.write_parquet(health_path)
    income_df.write_parquet(income_path)
    labour_df.write_parquet(labour_path)

    configs = [
        SourceConfig(
            name="health",
            filepath=health_path,
            id_col="patient_id",
            temporal_cols=[
                TemporalColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                CategoricalColConfig(col_name="birth_date", prefix="BIRTH", static=True),
            ],
        ),
        SourceConfig(
            name="income",
            filepath=income_path,
            id_col="person_id",
            temporal_cols=[
                TemporalColConfig(col_name="year", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="income_type", prefix="INCOME"),
                CategoricalColConfig(col_name="region", prefix="REG", static=True),
            ],
            continuous_cols=[
                ContinuousColConfig(col_name="amount", prefix="AMT", static=True)
            ],
        ),
        SourceConfig(
            name="labour",
            filepath=labour_path,
            id_col="person_id",
            temporal_cols=[
                TemporalColConfig(col_name="date", is_primary=True, drop_na=True)
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="status", prefix="STATUS"),
                CategoricalColConfig(
                    col_name="residence_region", prefix="REGION", static=True
                ),
            ],
        ),
    ]

    return SourceCollection.from_configs(configs)


def test_cohort_accepts_source_variants(source_collection: SourceCollection, tmp_path: Path):
    """Cohort accepts Source, list[Source], and SourceCollection."""
    single_source = source_collection["health"]

    cohort_from_source = Cohort(
        name="single",
        sources=single_source,
        cache_dir=tmp_path / "cohorts",
    )
    assert len(cohort_from_source) == 4

    cohort_from_list = Cohort(
        name="list",
        sources=[source_collection["health"], source_collection["income"]],
        cache_dir=tmp_path / "cohorts",
    )
    assert len(cohort_from_list) == 4

    cohort_from_collection = Cohort(
        name="collection",
        sources=source_collection,
        cache_dir=tmp_path / "cohorts",
    )
    assert len(cohort_from_collection) == 4


def test_inclusion_criteria_intersection(source_collection: SourceCollection, tmp_path: Path):
    """Required criteria should intersect qualifying entity sets across sources."""
    cohort = Cohort(
        name="criteria",
        sources=source_collection,
        inclusion_criteria=[
            EntityInclusionCriteria(source_name="health", required=True, min_events=2),
            EntityInclusionCriteria(source_name="income", required=True, min_events=1),
        ],
        cache_dir=tmp_path / "cohorts",
    )

    assert cohort.entity_ids == {"P001", "P002"}


def test_optional_criteria_with_event_bounds_warn_and_do_not_filter(
    source_collection: SourceCollection, tmp_path: Path
):
    """Optional criteria with bounds should warn and leave the cohort unchanged."""
    with pytest.warns(UserWarning, match="required=False"):
        criteria = EntityInclusionCriteria(
            source_name="health",
            required=False,
            min_events=99,
            max_events=100,
        )

    cohort = Cohort(
        name="optional-bounds",
        sources=source_collection,
        inclusion_criteria=[criteria],
        cache_dir=tmp_path / "cohorts",
    )

    assert cohort.entity_ids == {"P001", "P002", "P003", "P004"}


def test_unknown_optional_criteria_source_raises(
    source_collection: SourceCollection, tmp_path: Path
):
    """Unknown sources should fail even when the criteria is optional."""
    with pytest.raises(KeyError, match="unknown source"):
        Cohort(
            name="unknown-source",
            sources=source_collection,
            inclusion_criteria=[
                EntityInclusionCriteria(source_name="missing", required=False)
            ],
            cache_dir=tmp_path / "cohorts",
        )


def test_build_entities_table_with_static_columns(source_collection: SourceCollection, tmp_path: Path):
    """Entities table should contain entity_id and static columns from all sources."""
    cohort = Cohort(
        name="entities",
        sources=source_collection,
        cache_dir=tmp_path / "cohorts",
    )

    entity_df = cohort.build_entities_table()

    assert "entity_id" in entity_df.columns
    assert "health__birth_date" in entity_df.columns
    assert "income__amount" in entity_df.columns
    assert "income__region" in entity_df.columns
    assert "labour__residence_region" in entity_df.columns
    assert entity_df.height == 4

    expected_ids = ["P001", "P002", "P003", "P004"]
    assert entity_df.get_column("entity_id").to_list() == expected_ids


def test_entity_cache_artifacts_are_written(source_collection: SourceCollection, tmp_path: Path):
    """Building entities table should write parquet and metadata under data/cohorts/<name>."""
    cache_root = tmp_path / "data" / "cohorts"
    cohort = Cohort(name="cache-test", sources=source_collection, cache_dir=cache_root)

    cohort.build_entities_table()

    entities_path = cache_root / "cache-test" / "entities" / "entities_with_static.parquet"
    metadata_path = cache_root / "cache-test" / "entities" / "metadata.json"

    assert entities_path.exists()
    assert metadata_path.exists()


def test_split_build_and_cache_full_table(source_collection: SourceCollection, tmp_path: Path):
    """Split output should include entity_id, static columns, and split label."""
    cache_root = tmp_path / "data" / "cohorts"
    cohort = Cohort(name="split-test", sources=source_collection, cache_dir=cache_root)

    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=123)
    split_df = cohort.build_or_load_splits(split_cfg)

    assert "entity_id" in split_df.columns
    assert "health__birth_date" in split_df.columns
    assert "split" in split_df.columns
    assert split_df.height == 4
    assert set(split_df.get_column("split").unique().to_list()).issubset(
        {"train", "val", "test"}
    )

    split_hash = split_cfg.config_hash()
    split_path = (
        cache_root / "split-test" / "splits" / split_hash / "entities_split.parquet"
    )
    split_metadata = cache_root / "split-test" / "splits" / split_hash / "metadata.json"

    assert split_path.exists()
    assert split_metadata.exists()


def test_split_is_deterministic_for_same_seed(source_collection: SourceCollection, tmp_path: Path):
    """Identical split config should produce deterministic labels."""
    cohort = Cohort(name="deterministic", sources=source_collection, cache_dir=tmp_path)

    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=42)
    a = cohort.build_or_load_splits(split_cfg, force_recompute=True)
    b = cohort.build_or_load_splits(split_cfg, force_recompute=True)

    assert a.select(["entity_id", "split"]).equals(b.select(["entity_id", "split"]))


def test_stratified_split_uses_static_column(source_collection: SourceCollection, tmp_path: Path):
    """Optional stratified split should work with a static column."""
    cohort = Cohort(name="stratified", sources=source_collection, cache_dir=tmp_path)

    split_cfg = CohortConfig(
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        seed=7,
        stratify_col="income__region",
    )
    split_df = cohort.build_or_load_splits(split_cfg, force_recompute=True)

    assert split_df.height == 4
    assert "split" in split_df.columns


def test_invalid_stratify_column_raises(source_collection: SourceCollection, tmp_path: Path):
    """Unknown stratification column should raise a clear error."""
    cohort = Cohort(name="bad-stratify", sources=source_collection, cache_dir=tmp_path)

    split_cfg = CohortConfig(
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        seed=7,
        stratify_col="missing_col",
    )

    with pytest.raises(ValueError, match="stratify_col"):
        cohort.build_or_load_splits(split_cfg, force_recompute=True)


def test_empty_cohort_produces_empty_tables(source_collection: SourceCollection, tmp_path: Path):
    """If criteria removes all entities, entity and split tables should be empty."""
    cohort = Cohort(
        name="empty",
        sources=source_collection,
        inclusion_criteria=[
            EntityInclusionCriteria(source_name="health", required=True, min_events=99)
        ],
        cache_dir=tmp_path,
    )

    entity_df = cohort.build_entities_table(force_recompute=True)
    split_df = cohort.build_or_load_splits(force_recompute=True)

    assert entity_df.height == 0
    assert split_df.height == 0
    assert "split" in split_df.columns


def test_empty_required_criteria_logs_warning(
    source_collection: SourceCollection, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Empty cohorts from required criteria should emit a warning."""
    caplog.set_level("WARNING", logger="Cohort")

    cohort = Cohort(
        name="empty-warning",
        sources=source_collection,
        inclusion_criteria=[
            EntityInclusionCriteria(source_name="health", required=True, min_events=99)
        ],
        cache_dir=tmp_path / "cohorts",
    )

    assert len(cohort) == 0
    assert any(
        "resolved to 0 entities" in record.message for record in caplog.records
    )


def test_missing_source_column_raises_clear_cohort_error(tmp_path: Path):
    """Missing source columns should raise a cohort-facing schema error."""
    df = pl.DataFrame(
        {
            "person_id": ["P001", "P002"],
            "event_date": ["2020-01-01", "2020-02-01"],
            "status": ["active", "inactive"],
        }
    )
    path = tmp_path / "broken.parquet"
    df.write_parquet(path)

    source = Source(
        SourceConfig(
            name="broken-source",
            filepath=path,
            id_col="person_id",
            temporal_cols=[
                TemporalColConfig(
                    col_name="event_date", is_primary=True, drop_na=True
                )
            ],
            categorical_cols=[
                CategoricalColConfig(col_name="status", prefix="STATUS"),
                CategoricalColConfig(
                    col_name="missing_region", prefix="REG", static=True
                ),
            ],
        )
    )

    with pytest.raises(
        SchemaError,
        match="Failed to resolve entity IDs for cohort 'broken-cohort'.*broken-source.*missing_region",
    ):
        Cohort(name="broken-cohort", sources=source, cache_dir=tmp_path)
