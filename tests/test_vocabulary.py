"""Tests for train-split vocabulary builder."""

from pathlib import Path

import polars as pl

from tab2seq.cohort import Cohort, CohortConfig, EntityInclusionCriteria
from tab2seq.config import TokenizerConfig
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
            "entity_id": ["E1", "E1", "E2", "E3", "E4"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
            ],
            "diagnosis": ["A", "B", "A", "C", "B"],
            "cost": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    labour_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E2", "E3", "E4"],
            "date": ["2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13"],
            "birthday": ["1990-01-01", "1980-05-01", "1975-07-01", "2000-10-10"],
            "native_language": ["english", "danish", "english", "french"],
            "weekly_hours": [37.0, 32.0, 40.0, 25.0],
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
            categorical_cols=[
                CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
            ],
            continuous_cols=[
                ContinuousColConfig(col_name="cost", prefix="COST", n_bins=10),
            ],
            timestamp_cols=[
                TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
            ],
        ),
        SourceConfig(
            name="labour",
            filepath=labour_path,
            id_col="entity_id",
            categorical_cols=[
                CategoricalColConfig(
                    col_name="native_language", prefix="LANG", static=True
                ),
            ],
            continuous_cols=[
                ContinuousColConfig(col_name="weekly_hours", prefix="HOURS"),
            ],
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


def test_vocabulary_fit_and_cache(tmp_path: Path):
    """Vocabulary is built from train split and persisted under cohort cache."""
    collection = _build_collection(tmp_path)
    cohort = Cohort(
        name="vocab-cohort",
        sources=collection,
        inclusion_criteria=[
            EntityInclusionCriteria(source_name="labour", required=True, min_events=1)
        ],
        cache_dir=tmp_path / "cohorts",
    )

    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=13)
    tok_cfg = TokenizerConfig()
    tok_cfg.vocabulary.min_token_count = 1
    vocab = Vocabulary(tok_cfg.vocabulary)
    vocab_df = vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    assert vocab_df.height > 0
    assert "pretty_token" in vocab_df.columns
    assert "[PAD]" in vocab.token2index
    assert any(tok.startswith("health__cost__BIN_") for tok in vocab.token2index)
    assert not any("birthday__DAYS_" in tok for tok in vocab.token2index)

    non_special = vocab_df.filter(pl.col("category") != "special")
    assert non_special.height > 0
    assert all(
        not tok.startswith(f"{src}__")
        for tok, src in zip(
            non_special.get_column("pretty_token").to_list(),
            non_special.get_column("source_name").to_list(),
        )
    )

    # Re-load from cache path
    vocab_reload = Vocabulary(tok_cfg.vocabulary)
    cached_df = vocab_reload.fit_from_cohort_train(cohort, split_cfg, force_recompute=False)
    assert cached_df.height == vocab_df.height


def test_vocabulary_train_only_behavior(tmp_path: Path):
    """Vocabulary tokens should come from train split entities only."""
    collection = _build_collection(tmp_path)
    cohort = Cohort(name="train-only", sources=collection, cache_dir=tmp_path / "cohorts")

    split_cfg = CohortConfig(train_frac=0.25, val_frac=0.25, test_frac=0.5, seed=99)
    split_df = cohort.build_or_load_splits(split_cfg, force_recompute=True)

    train_ids = set(
        split_df.filter(pl.col("split") == "train").get_column("entity_id").to_list()
    )

    tok_cfg = TokenizerConfig()
    tok_cfg.vocabulary.min_token_count = 1
    vocab = Vocabulary(tok_cfg.vocabulary)
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    # Build diagnosis values actually present in train set for comparison
    health = collection["health"].read_all()
    train_diag = set(
        health.filter(pl.col("entity_id").is_in(train_ids))
        .get_column("diagnosis")
        .cast(pl.Utf8)
        .to_list()
    )
    diag_tokens = {t for t in vocab.token2index if t.startswith("health__diagnosis__")}
    observed = {t.split("health__diagnosis__", 1)[1] for t in diag_tokens}

    assert observed.issubset(train_diag)
