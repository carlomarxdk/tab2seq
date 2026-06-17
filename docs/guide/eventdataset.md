# EventDataset

`EventDataset` encodes all events for all cohort entities into Parquet files, partitioned by split. It computes relative-date features, handles static token embedding, and exposes four record access patterns.

## Building and persisting

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

## Key config options

| Option | Description |
|--------|-------------|
| `reference_date` | Epoch for computing `time` (days since this date) |
| `threshold_date` | Cutoff date for events |
| `include_after_threshold` | Whether to include events after the threshold |
| `include_token_str` | Store human-readable token strings alongside IDs |
| `embed_static_in_events` | Prepend static tokens to each entity's event sequence |
| `relative_date_features` | List of `RelativeDateRule` for derived features (e.g. age) |

## Loading a saved dataset

```python
dataset_loaded = EventDataset.from_name(
    name="my_dataset_v1",
    registry_dir=cohort.cache_dir / "datasets",
)
```

This reloads the dataset from Parquet without requiring the original cohort, sources, or tokenizer.

## Access patterns

Four methods are available on any `EventDataset`:

```python
# Fetch a specific entity by ID (returns None if not in that split)
record = dataset_loaded.get_entity_record("E00003", split="train")

# Random sample
record = dataset_loaded.sample_entity_record(split="train", seed=7)

# Full iterator sweep
for record in dataset_loaded.iter_entity_records(split="train", shuffle=True, seed=42):
    pass

# Stateful one-at-a-time — remembers position across calls, returns None when exhausted
record = dataset_loaded.next_entity_record(split="val", shuffle=True, seed=0, reset=True)
while record is not None:
    record = dataset_loaded.next_entity_record(split="val", shuffle=True, seed=0)
```

All four methods accept a `format` parameter. See [Record Formats](formats.md) for details.
