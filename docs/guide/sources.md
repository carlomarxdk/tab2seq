# Sources

A `Source` describes one event table: its file path, entity ID column, timestamp, and feature columns.

## Column types

| Config class | Purpose |
|---|---|
| `CategoricalColConfig` | String/enum columns → token string |
| `ContinuousColConfig` | Numeric columns → binned token string |
| `TemporalColConfig` | Date/datetime columns |

## Static vs. dynamic columns

Columns marked `static=True` represent entity-level attributes that do not change per event (e.g. birthday, native language). They are:

- Excluded from the per-event token sequence
- Carried through to the cohort split table as entity attributes
- Available as input to `RelativeDateRule` for computing relative-date features

## Defining a Single Source

```python
## example with the Labor data
config_L = SourceConfig(
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
        ContinuousColConfig(col_name="weekly_hours", prefix="WEEKLY_HOURS")
    ],
    temporal_cols=[
        TemporalColConfig(col_name="date", is_primary=True, drop_na=True, col_type="datetime"),
        TemporalColConfig(col_name="birthday", is_primary=False, static=True, drop_na=True, col_type="datetime"),
    ],
    output_format="parquet",
)
source_L = Source(config=config_L)
```

## Defining via SourceCollection

```python
from tab2seq.source import (
    SourceCollection, SourceConfig,
    CategoricalColConfig, ContinuousColConfig, TemporalColConfig,
)

configs = [
    SourceConfig(
        name="health",
        filepath="data/health.parquet",
        id_col="entity_id",
        categorical_cols=[
            CategoricalColConfig(col_name="diagnosis", prefix="DIAG"),
        ],
        continuous_cols=[
            ContinuousColConfig(col_name="cost", prefix="COST", n_bins=20, strategy="quantile"),
        ],
        temporal_cols=[
            TemporalColConfig(col_name="date", is_primary=True, drop_na=True, col_type="datetime"),
        ],
    ),
]

collection = SourceCollection.from_configs(configs)
```

## Binning strategies for continuous columns

The `strategy` parameter on `ContinuousColConfig` controls how bin edges are computed during vocabulary fitting:

- `"quantile"` — equal-frequency bins (robust to skewed distributions)
- `"uniform"` — equal-width bins

Bin edges are fitted on train data only and serialised with the vocabulary.

## Prefixes

Each column config has a `prefix` that becomes the token string prefix, e.g. `DIAG_J18.1`, `COST_bin_3`. Prefixes must be unique within a source.

## Source with (Only Static Attributes)

If you have a source with only static attributes (aka no `temporal_cols`), you still can uase this object. Do not forget to assign `static=True` to every feature (otherwise you will get an error).
