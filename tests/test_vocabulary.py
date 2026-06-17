"""Tests for the vocabulary-first tokenization pipeline."""

from pathlib import Path

import polars as pl

from tab2seq.cohort import Cohort, CohortConfig, EntityInclusionCriteria
from tab2seq.source import (
    CategoricalColConfig,
    ContinuousColConfig,
    SourceCollection,
    SourceConfig,
    TemporalColConfig,
)
from tab2seq.tokenization import Vocabulary, VocabularyConfig


def _build_source_collection(tmp_path: Path) -> SourceCollection:
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
            temporal_cols=[
                TemporalColConfig(col_name="date", is_primary=True, drop_na=True),
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


def test_fit_from_cohort_train_builds_and_caches_vocabulary(tmp_path: Path):
    """fit_from_cohort_train builds and caches vocabulary artifacts."""
    collection = _build_source_collection(tmp_path)
    cohort = Cohort(
        name="vocab-cohort",
        sources=collection,
        inclusion_criteria=[
            EntityInclusionCriteria(source_name="labour", required=True, min_events=1)
        ],
        cache_dir=tmp_path / "cohorts",
    )

    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=13)
    vocab_cfg = VocabularyConfig(min_token_count=1)
    vocab = Vocabulary(vocab_cfg)
    vocab_df = vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    assert vocab_df.height > 0
    assert "pretty_token" in vocab_df.columns
    assert "[PAD]" in vocab.token2index
    assert any(tok.startswith("health__COST__BIN_") for tok in vocab.token2index)
    assert not any(tok.startswith("health__entity_id__") for tok in vocab.token2index)
    assert not any(tok.startswith("labour__entity_id__") for tok in vocab.token2index)
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
    vocab_reload = Vocabulary(vocab_cfg)
    cached_df = vocab_reload.fit_from_cohort_train(cohort, split_cfg, force_recompute=False)
    assert cached_df.height == vocab_df.height


def test_fit_from_cohort_train_uses_train_split_only(tmp_path: Path):
    """Vocabulary tokens are derived only from train-split entities."""
    collection = _build_source_collection(tmp_path)
    cohort = Cohort(name="train-only", sources=collection, cache_dir=tmp_path / "cohorts")

    split_cfg = CohortConfig(train_frac=0.25, val_frac=0.25, test_frac=0.5, seed=99)
    split_df = cohort.build_or_load_splits(split_cfg, force_recompute=True)

    train_ids = set(
        split_df.filter(pl.col("split") == "train").get_column("entity_id").to_list()
    )

    vocab = Vocabulary(VocabularyConfig(min_token_count=1))
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    # Build diagnosis values actually present in train set for comparison
    health = collection["health"].read_all()
    train_diag = set(
        health.filter(pl.col("entity_id").is_in(train_ids))
        .get_column("diagnosis")
        .cast(pl.Utf8)
        .to_list()
    )
    diag_tokens = {t for t in vocab.token2index if t.startswith("health__DIAG__")}
    observed = {t.split("health__DIAG__", 1)[1] for t in diag_tokens}

    assert observed.issubset(train_diag)


def _build_repeated_token_collection(tmp_path: Path) -> SourceCollection:
    repeated_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E1", "E1", "E2"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "diagnosis": ["A", "A", "A", "A"],
        }
    )
    repeated_path = tmp_path / "repeated.parquet"
    repeated_df.write_parquet(repeated_path)

    return SourceCollection.from_configs(
        [
            SourceConfig(
                name="repeated",
                filepath=repeated_path,
                id_col="entity_id",
                categorical_cols=[
                    CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
                ],
                temporal_cols=[
                    TemporalColConfig(col_name="date", is_primary=True, drop_na=True),
                ],
            )
        ]
    )


def test_count_mode_entity_unique_prunes_prolific_entity_repeats(tmp_path: Path):
    collection = _build_repeated_token_collection(tmp_path)
    cohort = Cohort(name="count-mode", sources=collection, cache_dir=tmp_path / "cohorts")
    split_cfg = CohortConfig(use_splits=False)

    overall_vocab = Vocabulary(
        VocabularyConfig(min_token_count=3, count_mode="overall")
    )
    overall_vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    entity_unique_vocab = Vocabulary(
        VocabularyConfig(min_token_count=3, count_mode="entity_unique")
    )
    entity_unique_vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    token = "repeated__DIAG__A"
    assert token in overall_vocab.token2index
    assert token not in entity_unique_vocab.token2index


def test_count_mode_entity_unique_uses_entity_frequency(tmp_path: Path):
    collection = _build_repeated_token_collection(tmp_path)
    cohort = Cohort(name="count-values", sources=collection, cache_dir=tmp_path / "cohorts")
    split_cfg = CohortConfig(use_splits=False)

    vocab = Vocabulary(VocabularyConfig(min_token_count=1, count_mode="entity_unique"))
    vocab_df = vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)

    row = vocab_df.filter(pl.col("token") == "repeated__DIAG__A")
    assert row.height == 1
    assert row.get_column("count").item() == 2
