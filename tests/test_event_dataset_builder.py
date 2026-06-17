"""Tests for event dataset construction from cohort splits and vocabulary."""

from pathlib import Path

import polars as pl

from tab2seq.cohort import Cohort, CohortConfig
from tab2seq.datasets import EventDataset, EventDatasetConfig, RelativeDateRule
from tab2seq.source import (
    CategoricalColConfig,
    SourceCollection,
    SourceConfig,
    TemporalColConfig,
)
from tab2seq.tokenization import Tokenizer, Vocabulary, VocabularyConfig


def _build_source_collection(tmp_path: Path) -> SourceCollection:
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
            temporal_cols=[
                TemporalColConfig(col_name="date", is_primary=True, drop_na=True),
            ],
        ),
        SourceConfig(
            name="labour",
            filepath=labour_path,
            id_col="entity_id",
            categorical_cols=[CategoricalColConfig(col_name="status", prefix="STATUS")],
            temporal_cols=[
                TemporalColConfig(col_name="date", is_primary=True, drop_na=True),
                TemporalColConfig(
                    col_name="birthday",
                    static=True,
                    origin="1970-01-01",
                    unit="days",
                ),
            ],
        ),
    ]

    return SourceCollection.from_configs(configs)


def _build_dataset_inputs(tmp_path: Path) -> tuple[Cohort, Tokenizer]:
    collection = _build_source_collection(tmp_path)
    cohort = Cohort(name="dataset-cohort", sources=collection, cache_dir=tmp_path / "cohorts")
    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=42)

    vocab = Vocabulary(VocabularyConfig())
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)
    tokenizer = Tokenizer(vocabulary=vocab)
    return cohort, tokenizer


def test_build_split_has_required_columns(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)

    builder = EventDataset(
        cohort=cohort,
        tokenizer=tokenizer,
        dataset_config=EventDatasetConfig(),
    )

    train_df = builder.build_split("train", force_recompute_splits=True)
    assert train_df.height > 0
    assert {"entity_id", "split", "source_name", "primary_timestamp", "token_ids"}.issubset(
        set(train_df.columns)
    )
    assert "token_str" in train_df.columns
    assert "primary_time" in train_df.columns
    assert not train_df.get_column("token_str").str.contains("__entity_id__").any()


def test_write_parquet_separate_static_default(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)

    builder = EventDataset(cohort=cohort, tokenizer=tokenizer)

    artifacts = builder.write_parquet(force_recompute_splits=True, embed_static_in_events=False)
    assert artifacts.metadata_path.exists()
    assert artifacts.static_path.exists()
    assert artifacts.split_paths

    first_split_path = next(iter(artifacts.split_paths.values()))
    split_df = pl.read_parquet(first_split_path)
    static_df = pl.read_parquet(artifacts.static_path)

    assert "labour__birthday" not in split_df.columns
    assert "labour__birthday" in static_df.columns


def test_embed_static_in_events_true(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    builder = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = builder.build_split("train", force_recompute_splits=True, embed_static_in_events=True)
    assert "labour__birthday" in train_df.columns


def test_relative_date_rule_age_years(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)

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
        tokenizer=tokenizer,
        dataset_config=dataset_cfg,
    )

    train_df = builder.build_split("train", force_recompute_splits=True)
    assert "age_years" in train_df.columns
    assert train_df.get_column("age_years").null_count() < train_df.height


def test_get_entity_record_returns_static_and_events(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)

    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(entity_id=entity_id, split="train")
    assert record is not None
    assert record["entity_id"] == entity_id
    assert record["split"] == "train"
    assert isinstance(record["static"], dict)
    assert record["static"]["entity_id"] == entity_id
    assert record["static"]["split"] == "train"
    assert "token_ids" in record["static"]
    assert isinstance(record["static"]["token_ids"], list)
    assert "token_str" in record["static"]
    assert isinstance(record["static"]["token_str"], str)
    assert "birthday" not in record["static"]["token_str"]
    assert isinstance(record["events"], list)
    assert len(record["events"]) > 0


def test_sample_entity_record_seed_is_deterministic(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    _ = dataset.build_split("train", force_recompute_splits=True)

    sample_a = dataset.sample_entity_record(split="train", seed=7)
    sample_b = dataset.sample_entity_record(split="train", seed=7)

    assert sample_a is not None
    assert sample_b is not None
    assert sample_a["entity_id"] == sample_b["entity_id"]


def test_iter_entity_records_covers_split_entities(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)

    expected = sorted(set(train_df.get_column("entity_id").to_list()))
    observed = sorted({r["entity_id"] for r in dataset.iter_entity_records(split="train")})
    assert observed == expected


def test_next_entity_record_sequence_and_reset(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
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


def test_primary_time_can_be_zero(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(
        cohort=cohort,
        tokenizer=tokenizer,
        dataset_config=EventDatasetConfig(reference_date="2020-01-01"),
    )

    train_df = dataset.build_split("train", force_recompute_splits=True)
    assert (
        train_df
        .filter(pl.col("primary_timestamp") == "2020-01-01")
        .select(pl.col("primary_time").eq(0).all())
        .item()
    )


def test_include_after_threshold_filters_events(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    # threshold on 2020-01-04: events on 2020-01-05+ have after_threshold=True
    dataset = EventDataset(
        cohort=cohort,
        tokenizer=tokenizer,
        dataset_config=EventDatasetConfig(threshold_date="2020-01-04"),
    )
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    all_events = dataset.get_entity_record(entity_id, split="train", include_after_threshold=True)
    pre_only = dataset.get_entity_record(entity_id, split="train", include_after_threshold=False)

    assert all_events is not None
    assert pre_only is not None
    assert len(pre_only["events"]) <= len(all_events["events"])

    # Frame format: no after_threshold=True rows when filtered
    rec_frame = dataset.get_entity_record(entity_id, split="train", format="frame", include_after_threshold=False)
    assert rec_frame is not None
    if "after_threshold" in rec_frame["events"].columns:
        assert not rec_frame["events"]["after_threshold"].any()


def test_max_events_right_censoring(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record_full = dataset.get_entity_record(entity_id, split="train")
    assert record_full is not None
    n_full = len(record_full["events"])

    record_censored = dataset.get_entity_record(
        entity_id, split="train", censoring="right", max_events=1
    )
    assert record_censored is not None
    assert len(record_censored["events"]) == min(1, n_full)
    # Right censoring keeps earliest — first event should match
    if n_full > 0:
        assert record_censored["events"][0]["primary_timestamp"] == record_full["events"][0]["primary_timestamp"]


def test_max_events_left_censoring(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record_full = dataset.get_entity_record(entity_id, split="train")
    assert record_full is not None
    n_full = len(record_full["events"])

    record_left = dataset.get_entity_record(
        entity_id, split="train", censoring="left", max_events=1
    )
    assert record_left is not None
    assert len(record_left["events"]) == min(1, n_full)
    # Left censoring keeps latest — last event should match
    if n_full > 0:
        assert record_left["events"][-1]["primary_timestamp"] == record_full["events"][-1]["primary_timestamp"]


def test_max_events_and_max_tokens_raises(tmp_path: Path):
    import pytest as _pytest
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    with _pytest.raises(ValueError, match="Specify either max_events or max_tokens"):
        dataset.get_entity_record(
            entity_id, split="train", censoring="right", max_events=2, max_tokens=10
        )


def test_max_tokens_limits_total_tokens(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(
        entity_id, split="train", censoring="right", max_tokens=3, include_sep=False
    )
    assert record is not None
    total_tokens = sum(len(e["token_ids"]) for e in record["events"])
    assert total_tokens <= 3


def test_tensor_token_str_populated(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True, include_token_str=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(entity_id, split="train", format="tensor")
    assert record is not None
    assert record["token_str"] is not None
    assert isinstance(record["token_str"], list)
    assert all(isinstance(s, str) for s in record["token_str"])
    assert len(record["token_str"]) == len(record["event_lengths"])


def test_tensor_token_str_none_without_include_token_str(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True, include_token_str=False)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(entity_id, split="train", format="tensor")
    assert record is not None
    assert record["token_str"] is None


def test_tensor_static_token_str_includes_cls(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record_cls = dataset.get_entity_record(entity_id, split="train", format="tensor", include_cls=True)
    assert record_cls is not None
    assert record_cls["static_token_str"].startswith("[CLS]")

    record_no_cls = dataset.get_entity_record(entity_id, split="train", format="tensor", include_cls=False)
    assert record_no_cls is not None
    assert not record_no_cls["static_token_str"].startswith("[CLS]")


def test_frame_token_lengths_column(tmp_path: Path):
    cohort, tokenizer = _build_dataset_inputs(tmp_path)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    train_df = dataset.build_split("train", force_recompute_splits=True)
    entity_id = train_df.get_column("entity_id").to_list()[0]

    record = dataset.get_entity_record(entity_id, split="train", format="frame")
    assert record is not None
    assert "token_lengths" in record["events"].columns
    lengths = record["events"]["token_lengths"].to_list()
    actual = [len(ids) for ids in record["events"]["token_ids"].to_list()]
    assert lengths == actual
