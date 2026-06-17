# Installation

## Requirements

- Python ≥ 3.11
- NumPy ≥ 2.0
- Polars ≥ 1.38
- Pydantic v2

## From PyPI

```bash
pip install tab2seq
```

## Development install

```bash
git clone https://github.com/carlomarxdk/tab2seq
cd tab2seq
pip install -e ".[dev]"
```

The `dev` extras add pytest, ruff, mypy, and coverage tooling.

## Verify

```python
import tab2seq
print(tab2seq.__version__)
```
