# Record Formats

All four `EventDataset` access methods accept a `format` parameter that controls the output structure.

| Format | Returns | Best for |
|--------|---------|----------|
| `"raw"` | Python dicts (one dict per event) | inspection, custom collation |
| `"frame"` | Polars DataFrames | filtering, feature analysis |
| `"tensor"` | Flat NumPy arrays + event lengths | custom PyTorch/JAX collation |
| `"padded_tensor"` | 2-D padded NumPy matrix + attention mask | direct DataLoader use |

## `raw` (default)

```python
record = dataset_loaded.sample_entity_record("train", seed=42, format="raw")
# record["entity_id"]  → str
# record["split"]      → "train" | "val" | "test"
# record["static"]     → {"entity_id": ..., "labour__birthday": ..., "token_ids": [...], ...}
# record["events"]     → list of dicts, one per event:
#   event["primary_timestamp"]  → "2015-01-01"
#   event["source_name"]        → "labour"
#   event["token_ids"]          → [105, 86, 98, 110, 3]
#   event["age_years"]          → 28
```

## `frame`

Returns Polars DataFrames — avoids `to_dicts()` overhead for downstream filtering or analysis.

```python
record = dataset_loaded.sample_entity_record("train", seed=7, format="frame")
# record["entity_id"]        → str
# record["static_token_ids"] → list[int]
# record["events"]           → polars.DataFrame with columns:
#   primary_timestamp, source_name, token_ids (list[i64]), age_years, ...
```

## `tensor`

Returns flat NumPy arrays. `token_ids` concatenates all events into a single 1-D array; use `event_lengths` to split them back per event. `temporal` stacks `time` and any relative-date features into a `[num_events, T]` float array.

Pass `include_cls=True` to prepend a `[CLS]` token and `include_sep=True` to insert `[SEP]` between events.

```python
record = dataset_loaded.sample_entity_record(
    "train", seed=7, format="tensor", include_cls=True, include_sep=True
)
# record["token_ids"]        → ndarray shape (total_tokens,)  — all events concatenated
# record["event_lengths"]    → ndarray shape (num_events,)    — tokens per event
# record["time"]             → ndarray shape (num_events,)    — days since reference_date
# record["temporal"]         → ndarray shape (num_events, T)  — time + rel-date features
# record["static_token_ids"] → list[int]

# Reconstruct per-event token lists
import numpy as np
per_event = np.split(record["token_ids"], np.cumsum(record["event_lengths"])[:-1])
```

## `padded_tensor`

Like `tensor` but produces a 2-D `[num_events, max_event_len]` matrix padded with `pad_id`. Drops directly into a PyTorch `DataLoader` without further collation.

```python
record = dataset_loaded.sample_entity_record(
    "train", seed=7, format="padded_tensor", pad_id=0
)
# record["token_ids"]        → ndarray shape (num_events, max_event_len)
# record["attention_mask"]   → bool ndarray shape (num_events, max_event_len)
# record["time"]             → ndarray shape (num_events,)
# record["static_token_ids"] → list[int]
```
