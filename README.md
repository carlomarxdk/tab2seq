# tab2seq

[![PyPI - Version](https://img.shields.io/pypi/v/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Status](https://img.shields.io/pypi/status/tab2seq)](https://pypi.org/project/tab2seq/)
[![GitHub License](https://img.shields.io/github/license/carlomarxdk/tab2seq)](https://github.com/carlomarxdk/tab2seq/blob/main/LICENSE)

**tab2seq** turns multi-source tabular event data (registries, EHR, financial records) into tokenized sequences ready for Transformer-based models: it generalizes the data processing pipeline from the [Life2Vec](https://github.com/SocialComplexityLab/life2vec) paper to arbitrary domains.

> [!WARNING]
> This is an **alpha** package. The core pipeline (Sources → Cohort → Vocabulary → EventDataset) is functional but the API is not yet stable. Documentation is incomplete.  Pin to a specific version if you depend on current behaviour. See [TODOs](#roadmap) to see what is implemented at this point.

## Why tab2seq?

Building a [Life2Vec](https://github.com/SocialComplexityLab/life2vec)-style pipeline from scratch requires solving the same problems every time: multi-source schema alignment, leakage-safe vocabulary fitting, deterministic splits, and efficient Parquet-backed sequence iteration. `tab2seq` handles all of this so you can focus on modeling:

- Work with multiple longitudinal data sources (registries, databases)
- Define and filter cohorts based on inclusion criteria
- Create deterministic train/val/test splits with static context
- Fit a vocabulary on training data only (no leakage)
- Produce tokenized, model-ready event sequences with time features
- Generate realistic synthetic data for development and testing

**Requires:** Python ≥ 3.11, Numpy ≥ 2.0,  Polars ≥ 1.38, Pydantic v2.

## Pipeline

```text
Sources → Cohort → Vocabulary → Tokenizer -> EventDataset → Model-ready Parquet
```

| Step | Class | What it does |
|------|-------|--------------|
| 1 | `Source` / `SourceCollection` | Schema declaration for each event table (categorical, continuous, temporal columns) |
| 2 | `Cohort` | Entity universe + inclusion criteria + deterministic train/val/test splits |
| 3 | `Vocabulary` / `Tokenizer`| Token mappings and bin edges fitted on **train split only** |
| 4 | `EventDataset` | Vectorized token-ID encoding, relative-date features, Parquet persistence |

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

### 2. Define Sources

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

> Columns marked `static=True` are carried through to the cohort split table as entity-level attributes (e.g. birthday, native language).

### 3. Build a Cohort and Splits

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

cohort.build_entities_table(force_recompute=True)
split_cfg = CohortConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
cohort.build_or_load_splits(split_cfg)
print(f"Cohort size: {len(cohort)} entities")
```

The split table contains one row per entity with the split label and all static columns.

### 4. Fit a Vocabulary (Train Split Only)

The vocabulary maps categorical values to token strings and bins continuous features—fitted exclusively on training entities to prevent leakage.

```python
from tab2seq.config import TokenizerConfig
from tab2seq.tokenization import Tokenizer, Vocabulary

tok_cfg = TokenizerConfig()
tok_cfg.vocabulary.min_token_count = 1
tok_cfg.vocabulary.max_vocab_size = 50_000
tok_cfg.vocabulary.count_mode = "entity_unique"  # or "overall"

vocab = Vocabulary(tok_cfg.vocabulary)
vocab.fit_from_cohort_train(cohort=cohort, split_config=split_cfg)
print(f"Vocabulary size: {vocab.vocab_df.height}")
```

`VocabularyConfig.count_mode` controls how token frequency is computed for
`min_token_count` filtering:

- `overall`: counts every token occurrence across all train events.
- `entity_unique`: counts each token at most once per entity.

Use `entity_unique` to reduce dominance from very prolific entities.

### 5. Build and Persist Tokenized Event Datasets

`EventDataset` produces one row per event with integer token IDs, time features, and optional derived columns.

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
        relative_date_features=[
            RelativeDateRule(
                source_static_column="labour__birthday",
                output_column="age_years",
                unit="years",
            ),
        ],
    ),
)

artifacts = dataset.write_parquet(force_recompute_splits=True)
print(artifacts.split_paths)
```

### 6. Load a Precomputed Dataset by Name

You can reload a saved dataset without rebuilding sources, cohort, or tokenizer.

```python
dataset_loaded = EventDataset.from_name(
    name=dataset_name,
    registry_dir=cohort.cache_dir / "datasets",
)

sample = dataset_loaded.sample_entity_record("train", seed=42)
print("Loaded-by-name sample entity:", sample["entity_id"] if sample else None)
```

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

The `format` parameter on `get_entity_record`, `sample_entity_record`,
`iter_entity_records`, and `next_entity_record` lets you choose output type:

- `raw`: Python dict payload (default)
- `frame`: Polars DataFrame payload
- `tensor`: NumPy tensor-style payload
- `padded_tensor`: NumPy payload with padded token matrix + attention mask

```python
# 1) get_entity_record -> DataFrame payload
record_frame = dataset.get_entity_record(
    entity_id="E123",
    split="train",
    format="frame",
)

# 2) sample_entity_record -> tensor payload
record_tensor = dataset.sample_entity_record(
    split="train",
    seed=42,
    format="tensor",
)

# 3) iter_entity_records -> raw dict payload
for record_raw in dataset.iter_entity_records(split="train", format="raw"):
    pass

# 4) next_entity_record -> padded tensor payload
record_padded = dataset.next_entity_record(
    split="train",
    reset=True,
    format="padded_tensor",
    pad_id=0,
)
```

## Synthetic Registries

`generate_synthetic_data` / `generate_synthetic_collections` create four registry-style tables with realistic temporal patterns, missing data, and cross-field correlations:

| Registry | Key columns |
|----------|------------|
| **health** | diagnosis, procedure, department, cost, length_of_stay |
| **income** | income_type, sector, income_amount |
| **labour** | status, occupation, weekly_hours, residence_region, birthday |
| **survey** | education_level, marital_status, self_rated_health, satisfaction_score |

## Development

```bash
pip install -e ".[dev]"
pytest                          # run tests
pytest --cov=tab2seq            # with coverage
black src/tab2seq tests         # format
ruff check src/tab2seq tests    # lint
```

## Roadmap

- [x] Synthetic datasets
- [x] `Source` / `SourceCollection`
- [x] `Cohort` + splits
- [x] `Vocabulary` (leakage-safe)
- [x] `Tokenizer` / `EventDataset`
- [x] Parquet persistence + caching
- [ ] Full Life2Vec / Life2Vec-Light preprocessing parity
- [ ] Subseting Cohorts for finetuning
- [ ] Example with the Tokenization and Transformer training
- [ ] Documentation site

## Citation

If you use `tab2seq`, please cite:

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

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/carlomarxdk/tab2seq/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carlomarxdk/tab2seq/discussions)