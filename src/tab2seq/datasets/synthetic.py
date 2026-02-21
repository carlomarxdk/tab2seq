"""Generate synthetic registry data for exploration and testing.

Produces realistic-looking dummy data for multiple registries
(health, income, labour, survey) with configurable size.

Usage as a module::

    from tab2seq.datasets import generate_synthetic_collections, generate_synthetic_data

    collection = generate_synthetic_collections(output_dir="data/dummy", n_entities=1000)

    # returns a ready-to-use SourceCollection
    health = collection["health"]
    lf = health.scan()

Usage from CLI::

    tab2seq generate-synthetic_collections --output-dir data/dummy --n-entities 1000
    
    # generates files and prints a summary of the created collection
    
    tab2seq generate-synthetic-data --output-dir data/dummy --n-entities 1000 --registries health income --file-format csv
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from tab2seq.source.config import SourceConfig
from tab2seq.source.collection import SourceCollection

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Valid constants and catalogues
# ------------------------------------------------------------------

VALID_REGISTRIES = {"health", "income", "labour", "survey"}
VALID_FILE_FORMATS = {"parquet", "csv"}

# ------------------------------------------------------------------
# Catalogue of realistic-looking values
# ------------------------------------------------------------------

_ICD_CODES = [
    "I21.0",
    "I21.9",
    "I25.1",
    "I50.0",  # cardiovascular
    "J18.1",
    "J44.1",
    "J96.0",  # respiratory
    "C34.1",
    "C50.9",
    "C18.0",  # oncology
    "E11.9",
    "E78.0",  # metabolic
    "M54.5",
    "S72.0",
    "K80.2",  # other
]

_PROCEDURES = [
    "PCI",
    "CABG",
    "XRAY",
    "CT_SCAN",
    "MRI",
    "BIOPSY",
    "CHEMO",
    "RADIATION",
    "DIALYSIS",
    "ENDOSCOPY",
    "ECHO",
    "SPIROMETRY",
    "BLOOD_TEST",
]

_DEPARTMENTS = [
    "cardiology",
    "pulmonology",
    "oncology",
    "neurology",
    "orthopedics",
    "gastroenterology",
    "nephrology",
    "emergency",
    "general_surgery",
    "internal_medicine",
]

_INCOME_TYPES = [
    "salary",
    "self_employment",
    "pension",
    "unemployment_benefit",
    "disability_benefit",
    "capital_gains",
    "rental_income",
]

_SECTORS = [
    "public",
    "private",
    "non_profit",
    "self_employed",
]

_OCCUPATIONS = [
    "healthcare",
    "education",
    "engineering",
    "finance",
    "retail",
    "construction",
    "IT",
    "manufacturing",
    "transport",
    "agriculture",
    "hospitality",
    "research",
]

_LABOUR_STATUSES = [
    "employed",
    "unemployed",
    "self_employed",
    "student",
    "retired",
    "parental_leave",
    "sick_leave",
]

_EDUCATION_LEVELS = [
    "primary",
    "lower_secondary",
    "upper_secondary",
    "short_cycle_tertiary",
    "bachelor",
    "master",
    "doctoral",
]

_MARITAL_STATUSES = ["single", "married", "divorced", "widowed", "cohabiting"]

_RESIDENCE_REGIONS = ["north", "south", "east", "west", "central", "island", "capital"]

_SIMPLE_RESPONSES = ["yes", "no", "maybe", "prefer_not_to_say"]
_SIMPLE_RESPONSES_WITH_NOISE = [
    "yes",
    "no",
    "maybe",
    "n/a",
    None,
    "prefer_not_to_say",
    "",
]
_LIKERT_SCALE = [1, 2, 3, 4, 5]

_BIRTHDAY_START = date(1950, 1, 1)
_BIRTHDAY_END = date(2005, 12, 31)
_BIRTHDAY_RANGE = (_BIRTHDAY_END - _BIRTHDAY_START).days

# ------------------------------------------------------------------
# Per-registry generators
# ------------------------------------------------------------------


def _generate_entity_ids(n_entities: int) -> list[str]:
    """Generate zero-padded entity IDs."""
    width = len(str(n_entities))
    return [f"E{str(i).zfill(width)}" for i in range(1, n_entities + 1)]


def _generate_health(
    entity_ids: list[str],
    seed: int = 942,
    start_date: date = date(2015, 1, 1),
    end_date: date = date(2024, 12, 31),
) -> pl.DataFrame:
    """Generate synthetic health registry data.

    Each entity gets a variable number of health events (0–15)
    with diagnosis codes, procedures, departments, and costs.
    """
    total_days = (end_date - start_date).days
    rows: dict[str, list] = {
        "entity_id": [],
        "date": [],
        "diagnosis": [],
        "procedure": [],
        "department": [],
        "cost": [],
        "length_of_stay": [],
    }

    rng = np.random.default_rng(seed)
    for eid in entity_ids:
        n_events = rng.poisson(lam=4)
        if n_events == 0:
            continue
        for _ in range(n_events):
            day_offset = rng.integers(0, total_days)
            rows["entity_id"].append(eid)
            rows["date"].append(start_date + timedelta(days=int(day_offset)))
            rows["diagnosis"].append(rng.choice(_ICD_CODES))
            rows["procedure"].append(rng.choice(_PROCEDURES))
            rows["department"].append(rng.choice(_DEPARTMENTS))
            rows["cost"].append(round(float(rng.lognormal(mean=7.0, sigma=1.5)), 2))
            rows["length_of_stay"].append(max(0, int(rng.exponential(scale=3.0))))

    return pl.DataFrame(rows).sort("entity_id", "date")


def _generate_income(
    entity_ids: list[str],
    seed: int = 942,
    start_year: int = 2015,
    end_year: int = 2024,
) -> pl.DataFrame:
    """Generate synthetic income registry data.

    Each entity gets yearly income records with type, sector,
    and amount.
    """

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "entity_id": [],
        "year": [],
        "income_type": [],
        "sector": [],
        "income_amount": [],
    }

    for eid in entity_ids:
        # primary income type tends to persist
        primary_type = rng.choice(_INCOME_TYPES[:4])
        primary_sector = rng.choice(_SECTORS)
        base_income = float(rng.lognormal(mean=12.0, sigma=0.6))

        for year in range(start_year, end_year + 1):
            if rng.random() < 0.05:
                continue  # missed year
            # small chance of income type change
            if rng.random() < 0.1:
                primary_type = rng.choice(_INCOME_TYPES)
            growth = 1 + rng.normal(0.02, 0.05)
            base_income = max(0, base_income * growth)

            rows["entity_id"].append(eid)
            rows["year"].append(date(year, 12, 31))
            rows["income_type"].append(primary_type)
            rows["sector"].append(primary_sector)
            rows["income_amount"].append(round(base_income, 2))

    return pl.DataFrame(rows).sort("entity_id", "year")


def _generate_labour(
    entity_ids: list[str],
    seed: int = 942,
    start_date: date = date(2015, 1, 1),
    end_date: date = date(2024, 12, 31),
) -> pl.DataFrame:
    """Generate synthetic labour registry data.

    Each entity gets quarterly status records with occupation
    and hours.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "entity_id": [],
        "date": [],
        "status": [],
        "occupation": [],
        "weekly_hours": [],
        "residence_region": [],
        "birthday": [],
    }

    for eid in entity_ids:
        status = rng.choice(_LABOUR_STATUSES[:3])
        occupation = rng.choice(_OCCUPATIONS)
        residence_region = rng.choice(_RESIDENCE_REGIONS)
        current = start_date
        birthday = _BIRTHDAY_START + timedelta(
            days=int(rng.integers(0, _BIRTHDAY_RANGE))
        )

        while current <= end_date:
            # occasional status transitions
            if rng.random() < 0.05:
                status = rng.choice(_LABOUR_STATUSES)
            if rng.random() < 0.03:
                occupation = rng.choice(_OCCUPATIONS)

            # residence region changes infrequently
            if rng.random() < 0.02:
                residence_region = rng.choice(_RESIDENCE_REGIONS)

            hours = (
                round(float(rng.normal(37, 5)), 1)
                if status in ("employed", "self_employed")
                else 0.0
            )

            rows["entity_id"].append(eid)
            rows["date"].append(current)
            rows["status"].append(status)
            rows["occupation"].append(occupation)
            rows["weekly_hours"].append(max(0.0, hours))
            rows["residence_region"].append(residence_region)
            rows["birthday"].append(birthday)
            # advance by ~1-3 months
            advance_by = rng.integers(1, 4)
            month = current.month + advance_by
            year = current.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            current = date(year, month, 1)

    return pl.DataFrame(rows).sort("entity_id", "date")


def _generate_survey(
    entity_ids: list[str],
    seed: int = 942,
    start_year: int = 2016,
    end_year: int = 2024,
    survey_interval_years: int = 2,
) -> pl.DataFrame:
    """Generate synthetic survey data.

    Entities participate in periodic surveys with education,
    marital status, and satisfaction scores.
    """

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "entity_id": [],
        "survey_date": [],
        "education_level": [],
        "marital_status": [],
        "self_rated_health": [],
        "satisfaction_score": [],
        "question_1_response": [],
        "question_2_response": [],
        "question_3_response": [],
    }

    survey_years = list(range(start_year, end_year + 1, survey_interval_years))

    for eid in entity_ids:
        # ~60% participation rate per wave
        education = rng.choice(a = _EDUCATION_LEVELS)
        marital = rng.choice(a = _MARITAL_STATUSES)

        question_1_response = rng.choice(a = _SIMPLE_RESPONSES)
        # numpy does not support None types in choice, so we are selecting an index instead
        question_2_response_id = rng.choice(a = len(_SIMPLE_RESPONSES_WITH_NOISE))
        question_2_response = _SIMPLE_RESPONSES_WITH_NOISE[question_2_response_id]
        question_3_response = rng.choice(a = _LIKERT_SCALE)

        for year in survey_years:
            if rng.random() > 0.6:
                continue

            # education can progress
            if rng.random() < 0.05:
                idx = _EDUCATION_LEVELS.index(education)
                if idx < len(_EDUCATION_LEVELS) - 1:
                    education = _EDUCATION_LEVELS[idx + 1]

            if rng.random() < 0.08:
                marital = rng.choice(a = _MARITAL_STATUSES)

            if rng.random() < 0.5:
                question_1_response = rng.choice(a = _SIMPLE_RESPONSES)
            if rng.random() < 0.01:
                question_2_response_id = rng.choice(a = len(_SIMPLE_RESPONSES_WITH_NOISE))
                question_2_response = _SIMPLE_RESPONSES_WITH_NOISE[question_2_response_id]
            if rng.random() < 0.3:
                question_3_response = rng.choice(a = _LIKERT_SCALE)

            day = rng.integers(1, 28)
            month = rng.integers(3, 11)

            rows["entity_id"].append(eid)
            rows["survey_date"].append(date(year, int(month), int(day)))
            rows["education_level"].append(education)
            rows["marital_status"].append(marital)
            rows["self_rated_health"].append(int(rng.integers(1, 6)))
            rows["satisfaction_score"].append(round(float(rng.normal(6.5, 1.8)), 1))
            rows["question_1_response"].append(question_1_response)
            rows["question_2_response"].append(question_2_response)
            rows["question_3_response"].append(question_3_response)

    return pl.DataFrame(rows).sort("entity_id", "survey_date")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

#: Registry names and their generators + column specs
REGISTRY_SPECS: dict[str, dict] = {
    "health": {
        "generator": _generate_health,
        "entity_id_col": "entity_id",
        "timestamp_cols": ["date"],
        "categorical_cols": ["diagnosis", "procedure", "department"],
        "continuous_cols": ["cost", "length_of_stay"],
    },
    "income": {
        "generator": _generate_income,
        "entity_id_col": "entity_id",
        "timestamp_cols": ["year"],
        "categorical_cols": ["income_type", "sector"],
        "continuous_cols": ["income_amount"],
    },
    "labour": {
        "generator": _generate_labour,
        "entity_id_col": "entity_id",
        "timestamp_cols": ["date", "birthday"],
        "categorical_cols": ["status", "occupation", "residence_region"],
        "continuous_cols": ["weekly_hours"],
    },
    "survey": {
        "generator": _generate_survey,
        "entity_id_col": "entity_id",
        "timestamp_cols": ["survey_date"],
        "categorical_cols": [
            "education_level",
            "marital_status",
            "question_1_response",
            "question_2_response",
            "question_3_response",
        ],
        "continuous_cols": ["self_rated_health", "satisfaction_score"],
    },
}


def generate_synthetic_data(
    output_dir: str | Path = "data/synthetic",
    n_entities: int = 1000,
    seed: int = 742,
    registries: list[str] | None = None,
    file_format: str = "parquet",
) -> dict[str, Path]:
    """Generate synthetic registry data and write to disk.

    Writes one file per registry to ``output_dir`` in the specified format.

    Args:
        output_dir (str | Path): Directory where files will be written.
        n_entities (int): Number of unique entities to generate.
        seed (int): Random seed for reproducibility.
        registries (list[str] | None): Subset of registries to generate. Defaults to all
            available (health, income, labour, survey).
        file_format (str): Format of the output files. Must be 'parquet' or 'csv'.
    Returns:
        dict[str, Path]: Dictionary mapping registry names to the paths of the generated files.

    Example::

        from tab2seq.datasets import generate_synthetic_data

        data_paths = generate_synthetic_data(n_entities=500, file_format="parquet")
        health = pl.read_parquet(data_paths["health"])
    """
    if file_format not in VALID_FILE_FORMATS:
        msg = f"file_format must be one of {VALID_FILE_FORMATS}, got '{file_format}'"
        raise ValueError(msg)

    if registries is not None:
        unknown = set(registries) - VALID_REGISTRIES
        if unknown:
            available = ", ".join(VALID_REGISTRIES)
            msg = f"Unknown registries: {', '.join(unknown)}. Available: {available}"
            raise ValueError(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    entity_ids = _generate_entity_ids(n_entities)

    names = registries or list(REGISTRY_SPECS)
    paths: dict[str, Path] = {}

    for name in names:
        if name not in REGISTRY_SPECS:
            available = ", ".join(REGISTRY_SPECS)
            msg = f"Unknown registry '{name}'. Available: {available}"
            raise ValueError(msg)

        spec = REGISTRY_SPECS[name]
        generator = spec["generator"]

        logger.info("Generating '%s' for %d entities...", name, n_entities)
        df: pl.DataFrame = generator(entity_ids, rng.integers(0, 2**31))

        path = output_dir / f"{name}.{file_format}"
        if file_format == "parquet":
            df.write_parquet(path)
        else:
            df.write_csv(path)
        paths[name] = path
        logger.info("Wrote %s (%d rows)", path, df.height)
    return paths


def generate_synthetic_collections(
    output_dir: str | Path = "data/synthetic",
    n_entities: int = 1000,
    seed: int = 742,
    registries: list[str] | None = None,
    file_format: str = "parquet",
) -> SourceCollection:
    """Generate synthetic registry data and return a ready-to-use collection.

    Writes one file per registry to ``output_dir`` in the specified format and returns
    a :class:`~tab2seq.source.SourceCollection` pointing at them.

    Args:
        output_dir (str | Path): Directory where files will be written.
        n_entities (int): Number of unique entities to generate.
        seed (int): Random seed for reproducibility.
        registries (list[str] | None): Subset of registries to generate. Defaults to all
            available (health, income, labour, survey).
        file_format (str): Format of the output files. Must be 'parquet' or 'csv'.

    Returns:
        A :class:`SourceCollection` with one :class:`Source` per registry.

    Example::

        from tab2seq.datasets import generate_synthetic

        collection = generate_synthetic_collections(n_entities=500)
        health = collection["health"]
        print(health.scan().collect().head())
    """
    if file_format not in VALID_FILE_FORMATS:
        msg = f"file_format must be one of {VALID_FILE_FORMATS}, got '{file_format}'"
        raise ValueError(msg)

    if registries is not None:
        unknown = set(registries) - VALID_REGISTRIES
        if unknown:
            available = ", ".join(VALID_REGISTRIES)
            msg = f"Unknown registries: {', '.join(unknown)}. Available: {available}"
            raise ValueError(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    entity_ids = _generate_entity_ids(n_entities)

    names = registries or list(REGISTRY_SPECS)
    configs: list[SourceConfig] = []

    for name in names:
        if name not in REGISTRY_SPECS:
            available = ", ".join(REGISTRY_SPECS)
            msg = f"Unknown registry '{name}'. Available: {available}"
            raise ValueError(msg)

        spec = REGISTRY_SPECS[name]
        generator = spec["generator"]

        logger.info("Generating '%s' for %d entities...", name, n_entities)
        df: pl.DataFrame = generator(entity_ids, rng.integers(0, 2**31))

        path = output_dir / f"{name}.{file_format}"
        if file_format == "parquet":
            df.write_parquet(path)
        else:
            df.write_csv(path)
        logger.info("Wrote %s (%d rows)", path, df.height)

        configs.append(
            SourceConfig(
                name=name,
                filepath=path,
                entity_id_col=spec["entity_id_col"],
                timestamp_cols=spec["timestamp_cols"],
                categorical_cols=spec["categorical_cols"],
                continuous_cols=spec["continuous_cols"],
            )
        )

    return SourceCollection.from_configs(configs)
