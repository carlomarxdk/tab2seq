# tab2seq

[![PyPI - Version](https://img.shields.io/pypi/v/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Status](https://img.shields.io/pypi/status/tab2seq)](https://pypi.org/project/tab2seq/)
[![GitHub License](https://img.shields.io/github/license/carlomarxdk/tab2seq)](https://github.com/carlomarxdk/tab2seq/blob/main/LICENSE)

**tab2seq** turns multi-source tabular event data (registries, EHR, financial records) into tokenized sequences ready for Transformer-based models. It generalizes the data processing pipeline from the [Life2Vec](https://github.com/SocialComplexityLab/life2vec) paper to arbitrary domains.

!!! warning "Alpha software"
    The core pipeline (Sources → Cohort → Vocabulary → EventDataset) is functional but the API is not yet stable. Pin to a specific version if you depend on current behaviour.

## Why tab2seq?

Building a [Life2Vec](https://github.com/SocialComplexityLab/life2vec)-style pipeline from scratch requires solving the same problems every time: multi-source schema alignment, leakage-safe vocabulary fitting, deterministic splits, and efficient Parquet-backed sequence iteration. `tab2seq` handles all of this so you can focus on modeling:

- Work with multiple longitudinal data sources (registries, databases)
- Define and filter cohorts based on inclusion criteria
- Create deterministic train/val/test splits with static context
- Fit a vocabulary on training data only (no leakage)
- Produce tokenized, model-ready event sequences with time features
- Generate realistic synthetic data for development and testing

**Requires:** Python ≥ 3.11, NumPy ≥ 2.0, Polars ≥ 1.38, Pydantic v2.

## Pipeline

```text
Sources → Cohort → Vocabulary → Tokenizer → EventDataset → Model-ready Parquet
```

| Step | Class | What it does |
|------|-------|--------------|
| 1 | `Source` / `SourceCollection` | Schema declaration for each event table (categorical, continuous, temporal columns) |
| 2 | `Cohort` | Entity universe + inclusion criteria + deterministic train/val/test splits |
| 3 | `Vocabulary` / `Tokenizer` | Token mappings and bin edges fitted on **train split only** |
| 4 | `EventDataset` | Vectorized token-ID encoding, relative-date features, Parquet persistence |

## Quick install

```bash
pip install tab2seq
```

See [Installation](getting-started/installation.md) for full setup options and [Quick Start](getting-started/quickstart.md) to run the full pipeline end-to-end.

## Roadmap

- [x] Synthetic datasets
- [x] `Source` / `SourceCollection`
- [x] `Cohort` + splits
- [x] `Vocabulary` (leakage-safe)
- [x] `Tokenizer` / `EventDataset`
- [x] Parquet persistence + caching
- [ ] Full Life2Vec / Life2Vec-Light preprocessing parity
- [ ] Subsetting Cohorts for fine-tuning
- [ ] Example with Tokenization and Transformer training
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

Inspired by the data processing pipeline from [Life2Vec](https://github.com/SocialComplexityLab/life2vec) and [Life2Vec-Light](https://github.com/SocialComplexityLab/life2vec-light). Built with [Polars](https://polars.rs/) and [Pydantic](https://pydantic.dev/).
