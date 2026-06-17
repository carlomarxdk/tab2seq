# Pipeline Overview

```text
Sources → Cohort → Vocabulary → Tokenizer → EventDataset → Model-ready Parquet
```

`tab2seq` is structured as a sequential pipeline where each stage produces a persistent artifact that feeds the next.

## Stages

### 1. Sources

[`Source`](../api/tab2seq/source/index.md) / [`SourceCollection`](../api/tab2seq/source/index.md) declare the schema for each of your event tables — which columns are categorical, continuous, or temporal, and which are static entity-level attributes.

Sources are **lazy**: they hold config and a file path, and read data only when needed.

### 2. Cohort

[`Cohort`](../api/tab2seq/cohort/index.md) resolves a consistent entity universe across all sources, applies inclusion/exclusion criteria, and generates deterministic train/val/test splits. The split table is stored as Parquet and reloaded on subsequent runs.

### 3. Vocabulary

[`Vocabulary`](../api/tab2seq/tokenization/index.md) is fitted **on train entities only** to prevent leakage. It maps categorical values to token strings and learns bin edges for continuous features. The fitted vocabulary is serialised so it can be reloaded without the raw data.

### 4. Tokenizer

[`Tokenizer`](../api/tab2seq/tokenization/index.md) wraps a fitted `Vocabulary` and applies it to produce integer token IDs from raw feature values.

### 5. EventDataset

[`EventDataset`](../api/tab2seq/datasets/index.md) encodes all events into Parquet files partitioned by split. It computes relative-date features (e.g. age at event), handles static token embedding, and exposes four record access patterns.

## Design principles

**No leakage by construction.** The vocabulary is fitted only on the training split, and the dataset builder enforces this boundary.

**Parquet-first persistence.** Every stage caches its output as Parquet. Rebuilding is skipped automatically unless you pass `force_recompute=True`.

**Entity-centric records.** The dataset exposes one record per entity (not one row per event), which maps directly to how Transformer models consume sequences.
