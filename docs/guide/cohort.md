# Cohort & Splits

A `Cohort` resolves one consistent entity universe across all sources, applies inclusion criteria, and generates deterministic train/val/test splits.

## Building a cohort

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
```

## Inclusion criteria

Each `EntityInclusionCriteria` specifies:

- `source_name` — which source to apply the criterion to
- `required` — if `True`, entities without at least one event in this source are excluded
- `min_events` — minimum number of events an entity must have in this source

Entities that fail any `required` criterion are dropped from the cohort.

## Splits

```python
split_cfg = CohortConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
cohort.build_or_load_splits(split_cfg)
print(f"Cohort size: {len(cohort)} entities")
```

Splits are deterministic: given the same `seed`, the same entity will always land in the same split. The split table is cached as Parquet and reloaded on subsequent runs unless `force_recompute=True`.

The split table contains one row per entity with:

- `entity_id`
- `split` (`"train"` | `"val"` | `"test"`)
- All static columns from all sources (prefixed with `source_name__`)
