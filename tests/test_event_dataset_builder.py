"""Tests for event dataset builder."""

from pathlib import Path

import polars as pl

from tab2seq.cohort import Cohort, CohortConfig
from tab2seq.config import TokenizerConfig
from tab2seq.datasets import EventDataset, EventDatasetConfig, RelativeDateRule
from tab2seq.source import (
    CategoricalColConfig,
    ContinuousColConfig,
    SourceCollection,
    SourceConfig,
    TimestampColConfig,
)
from tab2seq.tokenization import Vocabulary


def _build_collection(tmp_path: Path) -> SourceCollection:
    health_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E1", "E2", "E3"],
            "date": ["2020-01-01", "2020-01-05", "2020-02-01", "2020-02-03"],
            "diagnosis": ["A", "B", "A", "C"],
            "cost": [10.0, 20.0, 30.0, 40.0],
        }
    )
    labour_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E2", "E3"],
            "date": ["2020-01-02", "2020-02-02", "2020-02-04"],
            "birthday": ["1990-01-01", "1980-06-01", "1975-07-01"],
            "status": ["employed", "unemployed", "employed"],
        }
    )

    health_path = tmp_path / "health.parquet"
    labour_path = tmp_path / "labour.parquet"
    health_df.write_parquet(health_path)
    labour_df.write_parquet(labour_path)

    configs = [
        SourceConfig(
            name="health",
            filepath=health_path,
            id_col="entity_id",
            categorical_cols=[CategoricalColConfig(col_name="diagnosis", prefix="DIAG")],
            continuous_cols=[ContinuousColConfig(col_name="cost", prefix="COST", n_bins=4)],
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
            ],
        ),
        SourceConfig(
            name="labour",
            filepath=labour_path,
            id_col="entity_id",
            categorical_cols=[CategoricalColConfig(col_name="status", prefix="STATUS")],
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
                TimestampColConfig(
                    col_name="birthday",
                    static=True,
                    origin="1970-01-01",
                    unit="days",
                ),
            ],
        ),
    ]

    return SourceCollection.from_configs(configs)


def _fitted_vocab_and_cohort(tmp_path: Path) -> tuple[Cohort, CohortConfig, Vocabulary]:
    collection = _build_collection(tmp_path)
    cohort = Cohort(name="dataset-cohort", sources=collection, cache_dir=tmp_path / "cohorts")
    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=42)

    tok_cfg = TokenizerConfig()
    vocab = Vocabulary(tok_cfg.vocabulary)
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)
    return cohort, split_cfg, vocab


def test_build_split_has_required_columns(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)

    builder = EventDataset(
        cohort=cohort,
        vocabulary=vocab,
        split_config=split_cfg,
        dataset_config=EventDatasetConfig(),
    )

    train_df = builder.build_split("train", force_recompute_splits=True)
    assert train_df.height > 0
    assert {"entity_id", "split", "source_name", "primary_timestamp", "token_ids"}.issubset(
        set(train_df.columns)
    )
    assert "token_str" in train_df.columns
    assert "primary_time" in train_df.columns


def test_write_parquet_separate_static_default(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)

    builder = EventDataset(
        cohort=cohort,
        vocabulary=vocab,
        split_config=split_cfg,
        dataset_config=EventDatasetConfig(embed_static_in_events=False),
    )

    artifacts = builder.write_parquet(force_recompute_splits=True)
    assert artifacts.metadata_path.exists()
    assert artifacts.static_path.exists()
    assert artifacts.split_paths

    first_split_path = next(iter(artifacts.split_paths.values()))
    split_df = pl.read_parquet(first_split_path)
    static_df = pl.read_parquet(artifacts.static_path)

    assert "labour__birthday" not in split_df.columns
    assert "labour__birthday" in static_df.columns


def test_embed_static_in_events_true(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)

    builder = EventDataset(
        cohort=cohort,
        vocabulary=vocab,
        split_config=split_cfg,
        dataset_config=EventDatasetConfig(embed_static_in_events=True),
    )

    train_df = builder.build_split("train", force_recompute_splits=True)
    assert "labour__birthday" in train_df.columns


def test_relative_date_rule_age_years(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)

    dataset_cfg = EventDatasetConfig(
        relative_date_features=[
            RelativeDateRule(
                source_static_column="labour__birthday",
                output_column="age_years",
                unit="years",
                floor_int=True,
            )
        ]
    )
    builder = EventDataset(
        cohort=cohort,
        vocabulary=vocab,
        split_config=split_cfg,
        dataset_config=dataset_cfg,
    )

    train_df = builder.build_split("train", force_recompute_splits=True)
    assert "age_years" in train_df.columns
    assert train_df.get_column("age_years").null_count() < train_df.height


def test_get_entity_record_returns_static_and_events(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)
    dataset = EventDataset(cohort=cohort, vocabulary=vocab, split_config=split_cfg)

    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(entity_id=entity_id, split="train")
    assert record is not None
    assert record["entity_id"] == entity_id
    assert record["split"] == "train"
    assert isinstance(record["static"], dict)
    assert isinstance(record["events"], list)
    assert len(record["events"]) > 0


def test_sample_entity_record_seed_is_deterministic(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)
    dataset = EventDataset(cohort=cohort, vocabulary=vocab, split_config=split_cfg)
    _ = dataset.build_split("train", force_recompute_splits=True)

    sample_a = dataset.sample_entity_record(split="train", seed=7)
    sample_b = dataset.sample_entity_record(split="train", seed=7)

    assert sample_a is not None
    assert sample_b is not None
    assert sample_a["entity_id"] == sample_b["entity_id"]


def test_iter_entity_records_covers_split_entities(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)
    dataset = EventDataset(cohort=cohort, vocabulary=vocab, split_config=split_cfg)
    train_df = dataset.build_split("train", force_recompute_splits=True)

    expected = sorted(set(train_df.get_column("entity_id").to_list()))
    observed = sorted({r["entity_id"] for r in dataset.iter_entity_records(split="train")})
    assert observed == expected


def test_next_entity_record_sequence_and_reset(tmp_path: Path):
    cohort, split_cfg, vocab = _fitted_vocab_and_cohort(tmp_path)
    dataset = EventDataset(cohort=cohort, vocabulary=vocab, split_config=split_cfg)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    n_entities = len(set(train_df.get_column("entity_id").to_list()))

    first = dataset.next_entity_record(split="train", shuffle=False, reset=True)
    assert first is not None

    seen = [first["entity_id"]]
    for _ in range(n_entities - 1):
        nxt = dataset.next_entity_record(split="train", shuffle=False)
        assert nxt is not None
        seen.append(nxt["entity_id"])

    assert len(set(seen)) == n_entities
    assert dataset.next_entity_record(split="train", shuffle=False) is None

    reset_first = dataset.next_entity_record(split="train", shuffle=False, reset=True)
    assert reset_first is not None
    assert reset_first["entity_id"] == first["entity_id"]
