"""Batch processing utilities for tokenized entities."""

from __future__ import annotations

from typing import Iterable

import polars as pl

from tab2seq.config import ProcessorConfig
from tab2seq.tokenization import Tokenizer


class BatchProcessor:
    """Convert grouped entity events into padded token id sequences."""

    def __init__(self, tokenizer: Tokenizer, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def process_entity(self, entity: tuple[object, pl.DataFrame]) -> dict[str, object]:
        """Process a single entity event frame into model-ready payload."""
        entity_id, events = entity
        token_ids = self.tokenizer.encode(events)
        token_ids = self.tokenizer.pad_sequence(token_ids, self.config.max_sequence_length)
        return {
            "entity_id": entity_id,
            "token_ids": token_ids,
            "length": len(token_ids),
        }

    def process_batch(self, entities: list[tuple[object, pl.DataFrame]]) -> list[dict[str, object]]:
        """Process a batch of entities."""
        return [self.process_entity(entity) for entity in entities]

    def process_stream(
        self,
        entities: Iterable[tuple[object, pl.DataFrame]],
    ) -> Iterable[list[dict[str, object]]]:
        """Process entities from a stream in batches."""
        batch: list[tuple[object, pl.DataFrame]] = []
        for entity in entities:
            batch.append(entity)
            if len(batch) >= self.config.batch_size:
                yield self.process_batch(batch)
                batch = []

        if batch:
            yield self.process_batch(batch)

    def close(self) -> None:
        """Close processor resources."""
        return None
