"""tab2seq - Transform tabular event data into sequences for transformer models."""

from importlib.metadata import version, PackageNotFoundError

from tab2seq.tokenization import Tokenizer, TokenizerConfig, Vocabulary, VocabularyConfig
from tab2seq.source import Source
from tab2seq.cohort import Cohort, CohortConfig

try:
    __version__ = version("tab2seq")
except PackageNotFoundError:
    # Package not installed (e.g. running from source without pip install -e .)
    __version__ = "unknown"

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
    "Vocabulary",
    "VocabularyConfig",
    "Source",
    "Cohort",
    "CohortConfig",
    "__version__",
]
