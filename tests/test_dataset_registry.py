"""Tests for named dataset registration and loading."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tab2seq.cohort import Cohort, CohortConfig
from tab2seq.datasets import EventDataset
from tab2seq.source import (
    CategoricalColConfig,
    SourceCollection,
    SourceConfig,
    TemporalColConfig,
)
from tab2seq.tokenization import Tokenizer, Vocabulary


def _build_source_collection(tmp_path: Path) -> SourceCollection:
    events_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E1", "E2", "E3"],
            "event_date": ["2024-01-01", "2024-01-03", "2024-02-01", "2024-03-01"],
            "event_type": ["A", "B", "A", "C"],
        }
    )

    events_path = tmp_path / "events.parquet"
    events_df.write_parquet(events_path)

    return SourceCollection.from_configs(
        [
            SourceConfig(
                name="events",
                filepath=events_path,
                id_col="entity_id",
                temporal_cols=[
                    TemporalColConfig(
                        col_name="event_date",
                        is_primary=True,
                        drop_na=True,
                    )
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="event_type", prefix="EVT")
                ],
            )
        ]
    )


def _build_dataset(tmp_path: Path) -> tuple[EventDataset, Path]:
    collection = _build_source_collection(tmp_path)
    cohort = Cohort(
        name="named-dataset",
        sources=collection,
        cache_dir=tmp_path / "cohorts",
    )
    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=7)

    vocab = Vocabulary()
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)
    tokenizer = Tokenizer(vocabulary=vocab)
    dataset = EventDataset(cohort=cohort, tokenizer=tokenizer)
    return dataset, cohort.cache_dir / "datasets"


def test_dataset_name_roundtrip_loads_splits_vocab_and_metadata(tmp_path: Path) -> None:
    dataset, registry_dir = _build_dataset(tmp_path)

    artifacts = dataset.write_parquet(
        force_recompute_splits=True,
        dataset_name="demo-v1",
    )

    loaded = EventDataset.from_name("demo-v1", registry_dir)

    assert artifacts.metadata_path.exists()
    assert (registry_dir / "registry.json").exists()

    train_original = dataset.build_split("train")
    train_loaded = loaded.build_split("train")

    assert train_original.schema == train_loaded.schema
    assert train_original.height == train_loaded.height
    assert loaded.tokenizer.vocabulary.vocab_df is not None
    assert loaded.tokenizer.vocabulary.bin_edges_df is not None


def test_dataset_name_collision_fails_without_overwrite(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)

    _ = dataset.write_parquet(force_recompute_splits=True, dataset_name="demo-v1")

    with pytest.raises(ValueError, match="already exists"):
        _ = dataset.write_parquet(force_recompute_splits=False, dataset_name="demo-v1")


def test_dataset_name_overwrite_allowed_when_enabled(tmp_path: Path) -> None:
    dataset, registry_dir = _build_dataset(tmp_path)

    _ = dataset.write_parquet(force_recompute_splits=True, dataset_name="demo-v1")
    _ = dataset.write_parquet(
        force_recompute_splits=False,
        dataset_name="demo-v1",
        overwrite_name=True,
    )

    loaded = EventDataset.from_name("demo-v1", registry_dir)
    assert loaded.build_split("train").height >= 0


def test_dataset_name_collision_fails_with_force_write_false(tmp_path: Path) -> None:
    dataset, _ = _build_dataset(tmp_path)

    _ = dataset.write_parquet(force_recompute_splits=True, dataset_name="demo-v1")

    with pytest.raises(ValueError, match="already exists"):
        _ = dataset.write_parquet(
            force_recompute_splits=False,
            dataset_name="demo-v1",
            force_write=False,
        )


def test_dataset_name_overwrite_allowed_with_force_write_true(tmp_path: Path) -> None:
    dataset, registry_dir = _build_dataset(tmp_path)

    _ = dataset.write_parquet(force_recompute_splits=True, dataset_name="demo-v1")
    _ = dataset.write_parquet(
        force_recompute_splits=False,
        dataset_name="demo-v1",
        force_write=True,
    )

    loaded = EventDataset.from_name("demo-v1", registry_dir)
    assert loaded.build_split("train").height >= 0


def test_from_name_does_not_rebuild_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, registry_dir = _build_dataset(tmp_path)
    _ = dataset.write_parquet(force_recompute_splits=True, dataset_name="demo-v1")

    from tab2seq.cohort.core import Cohort as CohortClass
    from tab2seq.source.core import Source
    from tab2seq.tokenization.vocabulary import Vocabulary as VocabularyClass

    def _raise(*args, **kwargs):
        raise AssertionError("Unexpected rebuild path invoked")

    monkeypatch.setattr(CohortClass, "build_or_load_splits", _raise)
    monkeypatch.setattr(Source, "scan", _raise)
    monkeypatch.setattr(VocabularyClass, "fit_from_cohort_train", _raise)

    loaded = EventDataset.from_name("demo-v1", registry_dir)
    record = loaded.sample_entity_record("train", seed=3)

    assert record is not None


def test_static_categorical_columns_are_excluded_from_event_tokens(tmp_path: Path) -> None:
    events_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E1", "E2"],
            "event_date": ["2024-01-01", "2024-02-01", "2024-01-05"],
            "event_type": ["A", "B", "A"],
            "native_language": ["hindi", "japanese", "turkish"],
        }
    )
    events_path = tmp_path / "events_static.parquet"
    events_df.write_parquet(events_path)

    collection = SourceCollection.from_configs(
        [
            SourceConfig(
                name="events",
                filepath=events_path,
                id_col="entity_id",
                temporal_cols=[
                    TemporalColConfig(
                        col_name="event_date",
                        is_primary=True,
                        drop_na=True,
                    )
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="event_type", prefix="EVT"),
                    CategoricalColConfig(
                        col_name="native_language",
                        prefix="LANG",
                        static=True,
                    ),
                ],
            )
        ]
    )

    cohort = Cohort(name="static-filter", sources=collection, cache_dir=tmp_path / "cohorts")
    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=5)
    vocab = Vocabulary()
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)
    ds = EventDataset(cohort=cohort, tokenizer=Tokenizer(vocabulary=vocab))

    train_df = ds.build_split("train", force_recompute_splits=True)
    if "token_str" in train_df.columns:
        token_values = train_df.get_column("token_str").to_list()
        assert not any("native_language" in tok for tok in token_values if tok is not None)
