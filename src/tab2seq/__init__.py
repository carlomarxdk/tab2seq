"""tab2seq - Transform tabular event data into sequences for transformer models."""

from importlib.metadata import version, PackageNotFoundError

from tab2seq.tokenization import Tokenizer, TokenizerConfig, Vocabulary, VocabularyConfig

try:
    __version__ = version("tab2seq")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Tokenizer",
    "TokenizerConfig",
    "Vocabulary",
    "VocabularyConfig",
    "__version__",
]
