# Synthetic Data

`tab2seq` ships with a synthetic data generator that produces four registry-style tables with realistic temporal patterns, missing data, and cross-field correlations. It is designed for testing and prototyping the pipeline without real data.

## Available registries

| Registry | Key columns |
|----------|------------|
| **health** | diagnosis, procedure, department, cost, length_of_stay |
| **income** | income_type, sector, income_amount |
| **labour** | status, occupation, weekly_hours, residence_region, birthday |
| **survey** | education_level, marital_status, self_rated_health, satisfaction_score |

## Generating data

```python
from tab2seq.datasets import generate_synthetic_data

data_paths = generate_synthetic_data(
    output_dir="synthetic_data",
    n_entities=10_000,
    seed=742,
    registries=["health", "labour", "survey", "income"],
)
# data_paths → {"health": Path(...), "labour": Path(...), ...}
```

## Generating a SourceCollection directly

```python
from tab2seq.datasets import generate_synthetic_collections

collection = generate_synthetic_collections(
    output_dir="synthetic_data",
    n_entities=10_000,
    seed=742,
)
```

This returns a ready-to-use `SourceCollection` wired to the generated files.
