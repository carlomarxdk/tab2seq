"""Registry for mapping dataset names to persisted dataset artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass
class DatasetRegistryEntry:
    """A named pointer to a persisted dataset artifact directory."""

    name: str
    dataset_hash: str
    dataset_dir: str
    metadata_path: str
    static_path: str
    split_hash: str | None
    vocab_hash: str | None
    vocab_dir: str | None
    cohort_name: str | None
    created_at_utc: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetRegistryEntry:
        """Create a typed registry entry from a JSON-compatible mapping."""
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this registry entry as a JSON-compatible mapping."""
        return asdict(self)


class DatasetRegistry:
    """JSON-backed dataset-name registry.

    The registry stores a mapping ``name -> DatasetRegistryEntry`` in a
    single JSON file located at ``{registry_dir}/registry.json``.
    """

    def __init__(self, registry_dir: Path) -> None:
        self._registry_dir = Path(registry_dir)
        self._registry_path = self._registry_dir / "registry.json"

    @property
    def path(self) -> Path:
        """Path to the underlying registry JSON file."""
        return self._registry_path

    def get(self, name: str) -> DatasetRegistryEntry | None:
        """Return one entry by name, or ``None`` if the name is unknown."""
        return self.list_entries().get(name)

    def list_entries(self) -> dict[str, DatasetRegistryEntry]:
        """Return all registry entries keyed by dataset name."""
        payload = self._load_payload()
        return {
            key: DatasetRegistryEntry.from_dict(value)
            for key, value in payload.items()
        }

    def register(self, entry: DatasetRegistryEntry, overwrite: bool = False) -> None:
        """Register a dataset name.

        Args:
            entry: Registry entry to persist.
            overwrite: Whether an existing name may be overwritten.

        Raises:
            ValueError: If the name already exists and overwrite is ``False``.
        """
        payload = self._load_payload()
        if entry.name in payload and not overwrite:
            raise ValueError(
                f"Dataset name '{entry.name}' already exists in registry "
                f"{self._registry_path}. Set overwrite=True to replace it."
            )

        self._registry_dir.mkdir(parents=True, exist_ok=True)
        payload[entry.name] = entry.to_dict()
        self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def now_utc_iso() -> str:
        """Return current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def _load_payload(self) -> dict[str, dict[str, Any]]:
        if not self._registry_path.exists():
            return {}
        raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid dataset registry format in {self._registry_path}."
            )
        return raw