"""Data loading helpers for event tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from tab2seq.config import LoaderConfig


class DataLoader:
    """Load tabular event data and group by entity."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()

    def load_chunks(self, path: str | Path) -> Iterator[pl.DataFrame]:
        """Yield chunks from a source file.

        Current implementation reads full file and slices by `chunk_size`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input format: {suffix}")

        chunk_size = self.config.chunk_size
        for start in range(0, df.height, chunk_size):
            yield df.slice(start, chunk_size)

    def group_by_entity(
        self,
        chunks: Iterator[pl.DataFrame],
    ) -> Iterator[tuple[object, pl.DataFrame]]:
        """Group event rows by configured entity column across all chunks."""
        entity_col = self.config.entity_id_column
        frames = list(chunks)
        if not frames:
            return

        for frame in frames:
            if entity_col not in frame.columns:
                raise ValueError(f"Entity ID column '{entity_col}' not found in input data")

        df = pl.concat(frames, how="vertical_relaxed")
        if self.config.timestamp_column in df.columns:
            df = df.sort([entity_col, self.config.timestamp_column])
        else:
            df = df.sort(entity_col)

        for value in df.get_column(entity_col).unique().to_list():
            yield value, df.filter(pl.col(entity_col) == value)

    def load_entities(self, path: str | Path) -> Iterator[tuple[object, pl.DataFrame]]:
        """Load file and yield grouped entity event frames."""
        return self.group_by_entity(self.load_chunks(path))
