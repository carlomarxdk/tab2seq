"""Tests for CLI module."""

import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from tab2seq.cli import init_config, main, process, validate_config


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_csv():
    """Create temporary CSV file with sample data."""
    data = pl.DataFrame(
        {
            "entity_id": [1, 1, 2, 2, 3],
            "timestamp": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03", "2020-01-01"],
            "event_type": ["A", "B", "A", "C", "B"],
            "value": [10, 20, 15, 25, 30],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data.write_csv(f.name)
        yield Path(f.name)
    try:
        Path(f.name).unlink()
    except FileNotFoundError:
        pass


def test_main_help(runner):
    """Test main command help."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "tab2seq" in result.output


def test_main_version(runner):
    """Test version flag."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    try:
        expected_version = version("tab2seq")
    except PackageNotFoundError:
        expected_version = "0.0.0+unknown"
    assert expected_version in result.output


def test_init_config(runner):
    """Test config initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "config.yaml"
        result = runner.invoke(init_config, ["--output", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()
        assert "Configuration saved" in result.output


def test_validate_config(runner):
    """Test config validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        runner.invoke(init_config, ["--output", str(config_path)])

        result = runner.invoke(validate_config, [str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()


def test_validate_config_invalid(runner):
    """Test validation of invalid config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("loader: [1, 2\n")
        config_path = Path(f.name)

    try:
        result = runner.invoke(validate_config, [str(config_path)])
        assert result.exit_code != 0
    finally:
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass


def test_process_basic(runner, sample_csv):
    """Test basic processing command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        result = runner.invoke(
            process,
            [
                str(sample_csv),
                "--output", str(output_dir),
                "--format", "csv",
            ],
        )

        assert result.exit_code == 0
        assert "Processing" in result.output
        assert output_dir.exists()


def test_process_with_config(runner, sample_csv):
    """Test processing with config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        output_dir = Path(tmpdir) / "output"

        # Create config
        runner.invoke(init_config, ["--output", str(config_path)])

        # Process with config
        result = runner.invoke(
            process,
            [
                str(sample_csv),
                "--config", str(config_path),
                "--output", str(output_dir),
            ],
        )

        assert result.exit_code == 0


def test_process_with_njobs(runner, sample_csv):
    """Test processing with parallel jobs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"

        result = runner.invoke(
            process,
            [
                str(sample_csv),
                "--output", str(output_dir),
                "--n-jobs", "2",
            ],
        )

        assert result.exit_code == 0
        assert "n_jobs=2" in result.output
