# tab2seq

[![PyPI - Version](https://img.shields.io/pypi/v/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tab2seq)](https://pypi.org/project/tab2seq/)
[![PyPI - Status](https://img.shields.io/pypi/status/tab2seq)](https://pypi.org/project/tab2seq/)
[![GitHub License](https://img.shields.io/github/license/carlomarxdk/tab2seq)](https://github.com/carlomarxdk/tab2seq/blob/main/LICENSE)

**tab2seq** adapts the Life2Vec data processing pipeline to make it easy to work with multi-source tabular event data for sequential modeling projects. Transform registry data, EHR records, and other event-based datasets into formats ready for Transformer and sequential deep learning models.

> [!WARNING]
> This is an alpha package. In the beta version, it will reimplement all the data-preprocessing steps of the [life2vec](https://github.com/SocialComplexityLab/life2vec) and [life2vec-light](https://github.com/carlomarxdk/life2vec-light) repos. See [TODOs](#todos) to see what is implemented at this point.

## About

This package extracts and generalizes the data processing patterns from the [Life2Vec](https://github.com/SocialComplexityLab/life2vec) project, making them reusable for similar research projects that need to:

- Work with multiple longitudinal data sources (registries, databases)
- Define and filter cohorts based on complex criteria
- Generate realistic synthetic data for development and testing
- Process large-scale tabular event data efficiently

Whether you're working with healthcare data, financial records, or any time-stamped event data, tab2seq provides the building blocks for preparing data for Life2Vec-style sequential models.

## Features

- **Multi-Source Data Management**: Handle multiple data sources (registries) with unified schema
- **Type-Safe Configuration**: Pydantic-based configuration with YAML support
- **Synthetic Data Generation**: Generate realistic dummy registry data for testing and exploration
- **Memory-Efficient Loading**: Chunked iteration and lazy loading with Polars
- **Schema Validation**: Automatic validation of entity IDs, timestamps, and column types
- **Cross-Source Operations**: Unified access and operations across multiple data sources

## Installation

```bash
# Basic installation
pip install tab2seq
```

## Quick Start

### Working with a Single Source

```python
from tab2seq.source import (
    Source, 
    SourceConfig, 
    SourceCollection, 
    CategoricalColConfig, 
    ContinuousColConfig, 
    TimestampColConfig
)

config = SourceConfig(
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
        ContinuousColConfig(col_name="length_of_stay", prefix="LOS", n_bins=20, strategy="quantile"),
    ],
    output_format="parquet",
    timestamp_cols=[
        TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
    ]
)

source = Source(config=config)

# Process and tokenize the columns
print("Number of unique IDs:", len(source.get_entity_ids()))
lf_health = source.process(cache=True)
lf_health.head()
```

### Working with Multiple Sources

```python
from tab2seq.source import SourceCollection, SourceConfig, CategoricalColConfig, ContinuousColConfig, TimestampColConfig

# Define your data sources
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
            ContinuousColConfig(col_name="length_of_stay", prefix="LOS", n_bins=20, strategy="quantile"),
        ],
        output_format="parquet",
        timestamp_cols=[
            TimestampColConfig(col_name="date", is_primary=True, drop_na=True)
        ]
    ),
    SourceConfig(
        name="labour",
        filepath="synthetic_data/labour.parquet",
        id_col="entity_id",
        categorical_cols=[
            CategoricalColConfig(col_name="status", prefix="STATUS"),
            CategoricalColConfig(col_name="occupation", prefix="OCC"),
            CategoricalColConfig(col_name="residence_region", prefix="REGION"),
        ],
        continuous_cols=[
            ContinuousColConfig(col_name="weekly_hours", prefix="WEEKLY_HOURS")
        ],
        output_format="parquet",
        timestamp_cols=[
            TimestampColConfig(col_name="date", is_primary=True, drop_na=True),
            TimestampColConfig(col_name="birthday", is_primary=False, drop_na=True),
        ],
    ),
]

# Create a source collection
collection = SourceCollection.from_configs(configs)

# Access individual sources
health = collection["health"]
df = health.read_all()

# Or iterate over all sources
for source in collection:
    print(f"{source.name}: {len(source.get_entity_ids())} entities")

# Cross-source operations
all_entity_ids = collection.get_all_entity_ids()
```

### Generating Synthetic Data

```python
from tab2seq.datasets import generate_synthetic_data
import polars as pl

# Generate synthetic registry data
data_paths = generate_synthetic_data(output_dir="synthetic_data", 
                                     n_entities=10000, 
                                     seed=742, 
                                     registries=["health", "labour", "survey", "income"],
                                     file_format="parquet")

lf_health = pl.read_parquet(data_paths["health"])
lf_health.head()
```

## Architecture

> [!warning]
> Work in progress!

**Available Registries:**

- **health**: Medical events with diagnoses (ICD codes), procedures, departments, costs, and length of stay
- **income**: Yearly income records with income type, sector, and amounts
- **labour**: Quarterly labour status with occupation, employment status, and residence
- **survey**: Periodic survey responses with education level, marital status, and satisfaction scores

All synthetic data includes realistic temporal patterns, missing data, and correlations between fields to mimic real-world registry data.

## Use Cases

- **Healthcare Research**: Transform electronic health records (EHR) into sequences for predictive modeling
- **Registry Data Processing**: Work with multiple event-based registries (health, income, labour, surveys)
- **Sequential Modeling**: Prepare multi-source data for Life2Vec, BEHRT, or other transformer-based models
- **Data Pipeline Development**: Use synthetic data to develop and test processing pipelines before working with sensitive real data
- **Multi-Source Analysis**: Combine and analyze data from multiple longitudinal sources with unified tooling

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=tab2seq --cov-report=html

# Format code
black src/tab2seq tests

# Lint code
ruff check src/tab2seq tests
```

## TODOs

- [x] Synthetic Datasets
- [x] `Source` implementation
- [ ] `Cohort` implementation
- [ ] `Cohort` and data splits
- [ ] `Tokenization` implementation
- [ ] `Vocabulary` implementation
- [x] Caching and chunking
- [ ] Documentation

## Citation

If you use this package in your research, please cite:

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
