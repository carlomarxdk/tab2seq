"""tab2seq - Transform tabular event data into sequences for transformer models."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tab2seq")
except PackageNotFoundError:
    # Package not installed (e.g. running from source without pip install -e .)
    __version__ = "unknown"