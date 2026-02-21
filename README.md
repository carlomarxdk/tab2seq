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

# Development installation
pip install -e ".[dev]"
```

## Quick Start

### Working with Multiple Data Sources

```python
from tab2seq.source import Source, SourceCollection, SourceConfig

# Define your data sources
configs = [
    SourceConfig(
        name="health",
        filepath="data/health.parquet",
        entity_id_col="patient_id",
        timestamp_col="date",
        categorical_cols=["diagnosis", "procedure", "department"],
        continuous_cols=["cost", "length_of_stay"],
    ),
    SourceConfig(
        name="income",
        filepath="data/income.parquet",
        entity_id_col="person_id",
        timestamp_col="year",
        categorical_cols=["income_type", "sector"],
        continuous_cols=["income_amount"],
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
from tab2seq.datasets import generate_synthetic_collections

# Generate synthetic registry data for testing
collection = generate_synthetic_collections(
    output_dir="data/dummy",
    n_entities=1000,
    seed=42
)

# Returns a ready-to-use SourceCollection
health = collection["health"]
print(health.read_all().head())
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
- [ ] Caching and chunking

## Citation

If you use this package in your research, please cite:

```bibtex
@software{tab2seq2024,
  author = {Savcisens, Germans},
  title = {tab2seq: Scalable Tabular to Sequential Data Processing},
  year = {2024},
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
- Built with [Polars](https://polars.rs/), [PyArrow](https://arrow.apache.org/docs/python/), [Pydantic](https://pydantic.dev/), and [Joblib](https://joblib.readthedocs.io/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/carlomarxdk/tab2seq).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/carlomarxdk/tab2seq/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carlomarxdk/tab2seq/discussions)
