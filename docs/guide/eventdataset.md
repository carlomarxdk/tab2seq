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

artifacts = dataset.write_parquet(
    dataset_name="my_dataset_v1",
    force_write=True,
    include_token_str=True,        # store human-readable token strings (default True)
    embed_static_in_events=False,  # keep static in a separate table (default False)
)
print(artifacts.dataset_dir)
```

## Key config options

| Option                     | Description                                                            |
|----------------------------|------------------------------------------------------------------------|
| `reference_date`           | Epoch for computing `primary_time` (days since this date per event)    |
| `threshold_date`           | Date used to flag post-threshold events                                |
| `include_after_threshold`  | If `True`, add an `after_threshold` boolean column to built events     |
| `relative_date_features`   | List of `RelativeDateRule` for per-event derived features (e.g. age)   |

## Build-time options

`include_token_str` and `embed_static_in_events` are passed directly to `build_split`, `build_all_splits`, or `write_parquet` — they are not stored in the config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_token_str` | `True` | Store human-readable token strings alongside IDs |
| `embed_static_in_events` | `False` | Left-join static columns into each event row (denormalised) |

```python
# Build without token strings to reduce size
train_df = dataset.build_split("train", include_token_str=False)

# Embed static columns directly in event rows
train_df = dataset.build_split("train", embed_static_in_events=True)
```

## Relative date features

`RelativeDateRule` computes a per-event derived value by comparing the event's timestamp to a static reference date. Unlike `TemporalColConfig(static=True)` (which stores a fixed date for the entity once), relative date features change per event:

```python
RelativeDateRule(
    source_static_column="labour__birthday",  # static column from any source
    output_column="age_years",                # name added to the event row
    unit="years",                             # "days", "weeks", "months", or "years"
    floor_int=True,                           # floor to nearest integer
)
```

The result (e.g. `age_years=28`) appears as an extra column in every event row and is stacked into the `temporal` matrix (column indices 1+) in tensor formats.

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

## Special tokens

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_cls` | `True` | Prepend `[CLS]` to the static token sequence |
| `include_sep` | `True` | Append `[SEP]` to each event's token sequence |
| `static_as_event` | `False` | Embed static tokens as event 0 instead of a separate field |

When `static_as_event=False` (default), `[CLS]` is prepended to `static_token_ids` and the static `token_str`. When `static_as_event=True`, `[CLS]` becomes the first token of event 0 and `[SEP]` is appended to it as well. `static_token_ids` is always populated regardless of this flag.

```python
# Static tokens kept separate (default)
record = dataset_loaded.get_entity_record("E1", split="train", include_cls=True, include_sep=True)
# record["static"]["token_ids"]  → [2, 105, 86]  ([CLS] + static tokens)
# record["events"][0]["token_ids"] → [98, 110, 3]  (event tokens + [SEP])

# Static tokens embedded as event 0
record = dataset_loaded.get_entity_record("E1", split="train", static_as_event=True)
# record["events"][0]["source_name"]  → "__static__"
# record["events"][0]["primary_time"] → 0
# record["static_token_ids"]          → still populated (without [SEP])
```

## Filtering and truncating events

All access methods support filtering and sequence length control:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_after_threshold` | `False` | When `False`, exclude events where `after_threshold=True` |
| `censoring` | `"none"` | `"right"` keeps earliest events; `"left"` keeps latest events |
| `max_events` | `None` | Maximum number of events per entity |
| `max_tokens` | `None` | Maximum total token count — whole events only, never partial |

`max_events` and `max_tokens` cannot be set together. When the entity is within the limit, `censoring` has no effect.

```python
# Keep only the 50 most recent events
for record in dataset.iter_entity_records("train", censoring="left", max_events=50):
    pass

# Limit to 512 tokens total, dropping events from the end
for record in dataset.iter_entity_records("train", censoring="right", max_tokens=512):
    pass

# Include post-threshold events (requires include_after_threshold=True in config)
record = dataset.get_entity_record("E1", split="train", include_after_threshold=True)
```
