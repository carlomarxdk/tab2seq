"""Tests for tokenizer module."""

from pathlib import Path

import polars as pl

from tab2seq import Tokenizer
from tab2seq.cohort import Cohort, CohortConfig
from tab2seq.config import TokenizerConfig
from tab2seq.source import (
    CategoricalColConfig,
    ContinuousColConfig,
    SourceCollection,
    SourceConfig,
    TemporalColConfig,
)
from tab2seq.tokenization import Vocabulary


def _build_source_collection(tmp_path: Path) -> SourceCollection:
    events_df = pl.DataFrame(
        {
            "entity_id": ["E1", "E1", "E2", "E3"],
            "wave": [1, 2, 1, 1],
            "event_date": [
                "2024-01-01",
                "2024-02-01",
                "2024-01-15",
                "2024-02-10",
            ],
            "event_type": ["A", "B", "A", "C"],
            "status": ["x", "y", "z", "x"],
            "cost": [10.0, 20.0, 30.0, 40.0],
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
                    TemporalColConfig(col_name="event_date", is_primary=True, drop_na=True),
                    TemporalColConfig(col_name="wave", col_type="ordinal"),
                ],
                categorical_cols=[
                    CategoricalColConfig(col_name="event_type", prefix="EVT"),
                    CategoricalColConfig(col_name="status", prefix="STATUS"),
                ],
                continuous_cols=[
                    ContinuousColConfig(col_name="cost", prefix="COST", n_bins=3)
                ],
            )
        ]
    )


def _build_fitted_tokenizer(tmp_path: Path, cfg: TokenizerConfig | None = None) -> Tokenizer:
    collection = _build_source_collection(tmp_path)
    cohort = Cohort(name="tok-cohort", sources=collection, cache_dir=tmp_path / "cohort")
    split_cfg = CohortConfig(train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=11)

    tok_cfg = cfg or TokenizerConfig()
    vocab = Vocabulary(tok_cfg.vocabulary)
    vocab.fit_from_cohort_train(cohort, split_cfg, force_recompute=True)
    return Tokenizer(vocabulary=vocab, config=tok_cfg)


def test_tokenizer_has_mapping_properties(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)

    assert tokenizer.vocab_size > 0
    assert tokenizer.token2index
    assert tokenizer.index2token
    assert tokenizer.config.cls_token in tokenizer.token2index


def test_encode_includes_cls_and_sep(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1", "E1"],
            "event_date": ["2024-01-01", "2024-02-01"],
            "wave": [1, 2],
            "event_type": ["A", "B"],
            "status": ["x", "y"],
            "cost": [10.0, 20.0],
        }
    )

    token_ids = tokenizer.encode(events, source_name="events")

    assert token_ids[0] == tokenizer.token2index[tokenizer.config.cls_token]
    assert token_ids[-1] == tokenizer.token2index[tokenizer.config.sep_token]


def test_encode_excludes_id_and_temporal_columns_by_default(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1", "E1"],
            "event_date": ["2024-01-01", "2024-02-01"],
            "wave": [1, 2],
            "event_type": ["A", "B"],
            "status": ["x", "y"],
            "cost": [10.0, 20.0],
        }
    )

    decoded = tokenizer.decode(tokenizer.encode(events, source_name="events"))

    assert not any(tok.startswith("events__entity_id__") for tok in decoded)
    assert not any(tok.startswith("events__event_date__") for tok in decoded)
    assert not any(tok.startswith("events__wave__") for tok in decoded)
    assert any(tok.startswith("events__event_type__") for tok in decoded)


def test_encode_respects_exclude_columns(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(
        tmp_path,
        cfg=TokenizerConfig(exclude_columns=["status"]),
    )
    events = pl.DataFrame(
        {
            "entity_id": ["E1", "E1"],
            "event_date": ["2024-01-01", "2024-02-01"],
            "wave": [1, 2],
            "event_type": ["A", "B"],
            "status": ["x", "y"],
            "cost": [10.0, 20.0],
        }
    )

    decoded = tokenizer.decode(tokenizer.encode(events, source_name="events"))

    assert not any(tok.startswith("events__status__") for tok in decoded)
    assert any(tok.startswith("events__event_type__") for tok in decoded)


def test_encode_with_explicit_columns_still_requires_vocab_tokens(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1"],
            "wave": [1],
            "event_type": ["A"],
        }
    )

    decoded = tokenizer.decode(
        tokenizer.encode(events, source_name="events", columns=["wave", "event_type"])
    )

    assert not any(tok.startswith("events__wave__") for tok in decoded)
    assert any(tok.startswith("events__event_type__") for tok in decoded)


def test_encode_with_explicit_unknown_column_is_filtered_out(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1"],
            "wave": [1],
        }
    )

    token_ids = tokenizer.encode(events, source_name="events", columns=["wave"])

    assert token_ids == [
        tokenizer.token2index[tokenizer.config.cls_token],
        tokenizer.token2index[tokenizer.config.sep_token],
    ]


def test_encode_frame_with_explicit_unknown_column_is_filtered_out(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1"],
            "wave": [1],
        }
    )

    encoded = tokenizer.encode_frame(events, source_name="events", columns=["wave"])

    assert encoded["token_ids"].to_list() == [None]


def test_encode_unknown_categorical_maps_to_unk(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E9"],
            "event_date": ["2024-03-01"],
            "wave": [3],
            "event_type": ["NEW_VALUE"],
            "status": ["x"],
            "cost": [10.0],
        }
    )

    token_ids = tokenizer.encode(events, source_name="events")
    unk_id = tokenizer.token2index[tokenizer.config.unk_token]
    assert unk_id in token_ids


def test_encode_skips_null_values(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame(
        {
            "entity_id": ["E1"],
            "event_date": ["2024-01-01"],
            "wave": [1],
            "event_type": [None],
            "status": ["x"],
            "cost": [None],
        }
    )

    decoded = tokenizer.decode(tokenizer.encode(events, source_name="events"))

    assert not any(tok.startswith("events__event_type__") for tok in decoded)


def test_decode_falls_back_to_unk_for_unknown_ids(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    decoded = tokenizer.decode([999999])
    assert decoded == [tokenizer.config.unk_token]


def test_pad_sequence_truncate_and_pad(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)

    seq = [
        tokenizer.token2index[tokenizer.config.cls_token],
        10,
        11,
        tokenizer.token2index[tokenizer.config.sep_token],
    ]
    truncated = tokenizer.pad_sequence(seq, max_length=3)
    assert len(truncated) == 3
    assert truncated[-1] == tokenizer.token2index[tokenizer.config.sep_token]

    padded = tokenizer.pad_sequence(seq, max_length=8)
    assert len(padded) == 8
    assert padded[-1] == tokenizer.token2index[tokenizer.config.pad_token]


def test_encode_unknown_source_returns_only_framing_tokens(tmp_path: Path):
    tokenizer = _build_fitted_tokenizer(tmp_path)
    events = pl.DataFrame({"event_type": ["A"], "status": ["x"]})

    token_ids = tokenizer.encode(events, source_name="missing_source")

    assert token_ids == [
        tokenizer.token2index[tokenizer.config.cls_token],
        tokenizer.token2index[tokenizer.config.sep_token],
    ]
