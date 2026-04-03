# tab2seq

[![PyPI - Version](https://img.shields.io/pypi/v/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Status](https://img.shields.io/pypi/status/tab2seq)](https://pypi.org/project/tab2seq/)
[![GitHub License](https://img.shields.io/github/license/carlomarxdk/tab2seq)](https://github.com/carlomarxdk/tab2seq/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1163020308.svg)](https://doi.org/10.5281/zenodo.18752504)

**tab2seq** adapts the Life2Vec data processing pipeline to make it easy to work with multi-source tabular event data for sequential modeling projects. Transform registry data, EHR records, and other event-based datasets into tokenized sequences ready for Transformer and sequential deep learning models.
The package reimplements the data-preprocessing steps of the [life2vec](https://github.com/SocialComplexityLab/life2vec) and [life2vec-light](https://github.com/carlomarxdk/life2vec-light) repos. 

> [!INFO]
> This is a **BETA** version of the package.

## About

This package extracts and generalizes the data processing patterns from the [Life2Vec](https://github.com/SocialComplexityLab/life2vec) project, making them reusable for similar research projects that need to:

- Work with multiple longitudinal data sources (registries, databases)
- Define and filter cohorts based on inclusion criteria
- Create deterministic train/val/test splits with static context
- Fit a vocabulary on training data only (no leakage)
- Produce tokenized, model-ready event sequences with time features
- Generate realistic synthetic data for development and testing

Whether you're working with healthcare data, financial records, or any time-stamped event data, tab2seq provides the building blocks for preparing data for Life2Vec-style sequential models.

## Pipeline Overview

```
Sources → Cohort → Vocabulary → EventDataset → Model-ready Parquet
```

1. **Sources** – Define one `SourceConfig` per event table (health visits, labour records, income, etc.). Each config declares which columns are categorical, continuous, or timestamps.
2. **Cohort** – Unite sources into a single entity universe, apply inclusion criteria, and split into train/val/test with deterministic seeds.
3. **Vocabulary** – Fit token mappings and continuous-feature bin edges on the *train split only* to prevent leakage.
4. **EventDataset** – Build tokenized event rows per split, derive relative-date features (e.g. age), and persist to Parquet with metadata.

## Features

- **Multi-Source Data Management**: Handle multiple data sources (registries) with unified schema
- **Cohort Construction**: Entity-level inclusion criteria across sources, deterministic splits, static-attribute propagation
- **Train-Only Vocabulary**: Token and bin-edge fitting restricted to training entities
- **Tokenized Event Datasets**: Vectorized token-ID encoding, relative-date features, Parquet persistence
- **Entity Record Access**: Iterator, random sample, and stateful `next()` retrieval patterns for downstream training loops
- **Type-Safe Configuration**: Pydantic-based configuration with YAML support
- **Synthetic Data Generation**: Generate realistic dummy registry data for testing and exploration
- **Memory-Efficient Loading**: Chunked iteration and lazy loading with Polars

## Installation

```bash
pip install tab2seq
```

## Quick Start

The full pipeline from raw data to model-ready sequences in five steps.

### 1. Generate Synthetic Data

```python
from tab2seq.datasets import generate_synthetic_data
import polars as pl

data_paths = generate_synthetic_data(
    output_dir="synthetic_data",
    n_entities=10_000,
    seed=742,
    registries=["health", "labour"],
)
pl.read_parquet(data_paths["health"]).head()
```

### 2. Define Sources

Each `Source` describes one event table: its file path, ID column, timestamp, and feature columns.

```python
from tab2seq.source import (
    Source, SourceCollection, SourceConfig,
    CategoricalColConfig, ContinuousColConfig, TimestampColConfig,
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
            ContinuousColConfig(col_name="cost", prefix="COST", n_bins=20),
            ContinuousColConfig(col_name="length_of_stay", prefix="LOS", n_bins=10),
        ],
        timestamp_cols=[
            TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
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
            ContinuousColConfig(col_name="weekly_hours", prefix="WEEKLY_HOURS", n_bins=10),
        ],
        timestamp_cols=[
            TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
            TimestampColConfig(col_name="birthday", static=True, drop_na=True),
        ],
    ),
]

collection = SourceCollection.from_configs(configs)

for source in collection:
    print(f"{source.name}: {len(source.get_entity_ids())} entities")
```

> Columns marked `static=True` are carried through to the cohort split table as entity-level attributes (e.g. birthday, native language).

### 3. Build a Cohort

A `Cohort` resolves one consistent entity universe across all sources, applies inclusion criteria, and generates deterministic train/val/test splits.

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

entities_df = cohort.build_entities_table(force_recompute=True)
print(f"Cohort size: {len(cohort)} entities")

split_cfg = CohortConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
split_df = cohort.build_or_load_splits(split_cfg, force_recompute=True)
split_df.head()
```

The split table contains one row per entity with the split label and all static columns.

### 4. Fit a Vocabulary (Train Only)

The vocabulary maps categorical values to token strings and bins continuous features—fitted exclusively on training entities to prevent leakage.

```python
from tab2seq.config import TokenizerConfig
from tab2seq.tokenization import Vocabulary

tok_cfg = TokenizerConfig()
tok_cfg.vocabulary.min_token_count = 1
tok_cfg.vocabulary.max_vocab_size = 50_000

vocab = Vocabulary(tok_cfg.vocabulary)
vocab_df = vocab.fit_from_cohort_train(
    cohort=cohort,
    split_config=split_cfg,
    force_recompute=True,
)
print(f"Vocabulary size: {vocab_df.height}")
```

### 5. Build Tokenized Event Datasets

`EventDataset` produces one row per event with integer token IDs, time features, and optional derived columns.

```python
from tab2seq.datasets import EventDataset, EventDatasetConfig, RelativeDateRule

dataset_cfg = EventDatasetConfig(
    reference_date="1970-01-01",
    threshold_date="2021-01-01",
    include_after_threshold=True,
    include_token_str=True,
    relative_date_features=[
        RelativeDateRule(
            source_static_column="labour__birthday",
            output_column="age_years",
            unit="years",
        ),
    ],
)

dataset = EventDataset(
    cohort=cohort,
    vocabulary=vocab,
    split_config=split_cfg,
    dataset_config=dataset_cfg,
)

# Inspect one split in memory
train_events = dataset.build_split("train", force_recompute_splits=True)
print(train_events.select(
    ["entity_id", "source_name", "primary_timestamp", "token_ids", "age_years"]
).head(5))

# Persist all splits + static table + metadata to Parquet
artifacts = dataset.write_parquet(force_recompute_splits=True)
print(artifacts.split_paths)
```

### Retrieving Entity Records

Three patterns for feeding records into a training loop:

```python
# Full iterator sweep
for record in dataset.iter_entity_records(split="train", shuffle=True, seed=42):
    # record = {"entity_id": ..., "split": ..., "static": {...}, "events": [...]}
    pass

# Random sample
record = dataset.sample_entity_record(split="train", seed=7)

# Stateful next() — remembers position across calls
record = dataset.next_entity_record(split="train", shuffle=True, seed=0, reset=True)
while record is not None:
    record = dataset.next_entity_record(split="train", shuffle=True, seed=0)
```

## Synthetic Registries

`generate_synthetic_data` / `generate_synthetic_collections` create four registry-style tables with realistic temporal patterns, missing data, and cross-field correlations:

| Registry | Key columns |
|----------|------------|
| **health** | diagnosis, procedure, department, cost, length_of_stay |
| **income** | income_type, sector, income_amount |
| **labour** | status, occupation, weekly_hours, residence_region, birthday |
| **survey** | education_level, marital_status, self_rated_health, satisfaction_score |

## Use Cases

- **Healthcare Research**: Transform electronic health records (EHR) into sequences for predictive modeling
- **Registry Data Processing**: Work with multiple event-based registries (health, income, labour, surveys)
- **Sequential Modeling**: Prepare multi-source data for Life2Vec, BEHRT, or other transformer-based models
- **Data Pipeline Development**: Use synthetic data to develop and test processing pipelines before working with sensitive real data


## TODOs

- [x] Synthetic Datasets
- [x] `Source` implementation
- [x] `Cohort` implementation
- [x] `Cohort` and data splits
- [x] `Tokenization` implementation
- [x] `Vocabulary` implementation
- [x] `EventDataset` builder
- [x] Caching and chunking
- [ ] Documentation

## Citation

If you use this package in your research, please cite:

```bibtex
@software{tab2seq2026,
  author = {Savcisens, Germans},
  title = {tab2seq: Scalable Tabular to Sequential Data Processing},
  year = {2026},
  url = {https://github.com/carlomarxdk/tab2seq}
}
```

And the original Life2Vec paper that inspired this work:

```bibtex
@article{savcisens2024using,
  title={Using sequences of life-events to predict human lives},
  author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust Hvas and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
  journal={Nature computational science},
  volume={4},
  number={1},
  pages={43--56},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```

## Acknowledgments

- Inspired by the data processing pipeline from [Life2Vec](https://github.com/SocialComplexityLab/life2vec) and [Life2Vec-Light](https://github.com/SocialComplexityLab/life2vec-light)
- Built with [Polars](https://polars.rs/) and [Pydantic](https://pydantic.dev/).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/carlomarxdk/tab2seq).

## License

MIT License: see [LICENSE](LICENSE) file for details.

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/carlomarxdk/tab2seq/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carlomarxdk/tab2seq/discussions)

