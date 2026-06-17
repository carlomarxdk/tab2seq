# Quick Start

The full pipeline from raw data to model-ready sequences in six steps.

## 1. Generate Synthetic Data

```python
from tab2seq.datasets import generate_synthetic_data
import polars as pl

data_paths = generate_synthetic_data(
    output_dir="synthetic_data",
    n_entities=10_000,
    seed=742,
    registries=["health", "labour", "survey", "income"],
)
pl.read_parquet(data_paths["health"]).head()
```

```text
shape: (5, 7)
┌───────────┬────────────┬───────────┬───────────┬──────────────────┬─────────┬────────────────┐
│ entity_id ┆ date       ┆ diagnosis ┆ procedure ┆ department       ┆ cost    ┆ length_of_stay │
│ str       ┆ date       ┆ str       ┆ str       ┆ str              ┆ f64     ┆ i64            │
╞═══════════╪════════════╪═══════════╪═══════════╪══════════════════╪═════════╪════════════════╡
│ E00001    ┆ 2016-09-15 ┆ J18.1     ┆ CABG      ┆ gastroenterology ┆ 7306.17 ┆ 2              │
│ E00001    ┆ 2017-05-25 ┆ E78.0     ┆ XRAY      ┆ neurology        ┆  138.65 ┆ 1              │
│ E00001    ┆ 2018-01-18 ┆ E78.0     ┆ MRI       ┆ general_surgery  ┆ 6704.59 ┆ 10             │
└───────────┴────────────┴───────────┴───────────┴──────────────────┴─────────┴────────────────┘
```

## 2. Define Sources

Each `Source` describes one event table: its file path, ID column, timestamp, and feature columns.

```python
from tab2seq.source import (
    Source, SourceCollection, SourceConfig,
    CategoricalColConfig, ContinuousColConfig, TemporalColConfig,
)

configs = [
    SourceConfig(
        name="health",
        filepath="synthetic_data/health.parquet",
        id_col="entity_id",
        categorical_cols=[
            CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
            CategoricalColConfig(col_name="procedure", prefix="PROC"),
            CategoricalColConfig(col_name="department", prefix="DEPT"),
        ],
        continuous_cols=[
            ContinuousColConfig(col_name="cost", prefix="COST", n_bins=20, strategy="quantile"),
            ContinuousColConfig(col_name="length_of_stay", prefix="LOS", n_bins=10, strategy="quantile"),
        ],
        temporal_cols=[
            TemporalColConfig(col_name="date", is_primary=True, drop_na=True, col_type="datetime"),
        ],
    ),
    SourceConfig(
        name="labour",
        filepath="synthetic_data/labour.parquet",
        id_col="entity_id",
        categorical_cols=[
            CategoricalColConfig(col_name="status", prefix="STATUS"),
            CategoricalColConfig(col_name="occupation", prefix="OCC"),
            CategoricalColConfig(col_name="residence_region", prefix="REGION"),
            CategoricalColConfig(col_name="native_language", prefix="LANG", static=True),
        ],
        continuous_cols=[
            ContinuousColConfig(col_name="weekly_hours", prefix="WEEKLY_HOURS", n_bins=10, strategy="uniform"),
        ],
        temporal_cols=[
            TemporalColConfig(col_name="date", is_primary=True, drop_na=True, col_type="datetime"),
            TemporalColConfig(col_name="birthday", static=True, drop_na=True, col_type="datetime"),
        ],
    ),
]

collection = SourceCollection.from_configs(configs)

for source in collection:
    print(f"{source.name}: {len(source.get_entity_ids())} entities")
```

!!! tip
    Columns marked `static=True` are carried through to the cohort split table as entity-level attributes (e.g. birthday, native language). See [Sources](../guide/sources.md) for details.

## 3. Build a Cohort and Splits

```python
from tab2seq.cohort import Cohort, CohortConfig, EntityInclusionCriteria

criteria = [
    EntityInclusionCriteria(source_name="health", required=False),
    EntityInclusionCriteria(source_name="labour", required=True, min_events=1),
]

cohort = Cohort(
    name="my_cohort",
    sources=collection,
    inclusion_criteria=criteria,
    cache_dir="data/cohorts",
)

cohort.build_entities_table(force_recompute=True)
split_cfg = CohortConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
cohort.build_or_load_splits(split_cfg)
print(f"Cohort size: {len(cohort)} entities")
```

## 4. Fit a Vocabulary (Train Split Only)

```python
from tab2seq.tokenization import Tokenizer, Vocabulary, VocabularyConfig

vocab = Vocabulary(
    config=VocabularyConfig(
        max_vocab_size=50_000,
        min_token_count=5,
        extra_tokens=["[DEATH]", "[RETIRED]"],
    )
)
vocab_df = vocab.fit_from_cohort_train(cohort=cohort, split_config=split_cfg)
print(f"Vocabulary size: {vocab_df.height}")
```

## 5. Build and Persist Tokenized Event Datasets

```python
from tab2seq.datasets import EventDataset, EventDatasetConfig, RelativeDateRule

dataset = EventDataset(
    cohort=cohort,
    tokenizer=Tokenizer(vocab),
    dataset_config=EventDatasetConfig(
        reference_date="1970-01-01",
        threshold_date="2021-01-01",
        include_after_threshold=True,
        include_token_str=True,
        embed_static_in_events=False,
        relative_date_features=[
            RelativeDateRule(
                source_static_column="labour__birthday",
                output_column="age_years",
                unit="years",
                floor_int=True,
            ),
        ],
    ),
)

artifacts = dataset.write_parquet(dataset_name="my_dataset_v1", force_write=True)
print(artifacts.dataset_dir)
```

## 6. Load and Read Records

```python
dataset_loaded = EventDataset.from_name(
    name="my_dataset_v1",
    registry_dir=cohort.cache_dir / "datasets",
)

record = dataset_loaded.sample_entity_record(split="train", seed=7)
```

See [Record Formats](../guide/formats.md) for all four output formats (`raw`, `frame`, `tensor`, `padded_tensor`).
