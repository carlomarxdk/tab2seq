# Vocabulary & Tokenizer

## Fitting a vocabulary

The vocabulary maps categorical values to token strings and learns bin edges for continuous features. It must be fitted **on train entities only** to prevent data leakage.

```python
from tab2seq.tokenization import Tokenizer, Vocabulary, VocabularyConfig

vocab = Vocabulary(
    config=VocabularyConfig(
        max_vocab_size=50_000,
        min_token_count=5,
        # Reserved tokens [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 [MASK]=4 are always included.
        extra_tokens=["[DEATH]", "[RETIRED]"],
    )
)
vocab_df = vocab.fit_from_cohort_train(cohort=cohort, split_config=split_cfg)
print(f"Vocabulary size: {vocab_df.height}")
```

## Reserved tokens

The following token IDs are always reserved regardless of `extra_tokens`:

| Token | ID |
|-------|-----|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[CLS]` | 2 |
| `[SEP]` | 3 |
| `[MASK]` | 4 |

## Count modes

`VocabularyConfig.count_mode` controls how token frequency is computed for `min_token_count` filtering:

- `"overall"` — counts every token occurrence across all train events
- `"entity_unique"` — counts each token at most once per entity

Use `"entity_unique"` to prevent very prolific entities from inflating token counts and keeping rare tokens that are actually rare across the population.

## Inspecting the fitted vocabulary

```python
# Column → prefix mapping per source
print(vocab.column_prefixes("health"))
# {'cost': 'COST', 'length_of_stay': 'LOS', 'diagnosis': 'DIAG', ...}

# Bin edges for a continuous column (fitted on train data only)
print(vocab.bin_edges_for("health", "cost"))
```

## Tokenizer

`Tokenizer` wraps a fitted `Vocabulary` and converts raw feature values to integer token IDs:

```python
from tab2seq.tokenization import Tokenizer

tokenizer = Tokenizer(vocab)
```

The tokenizer is passed directly to `EventDataset` and is not typically used standalone.
