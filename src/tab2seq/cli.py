"""Command-line entry points for tab2seq."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click

from tab2seq.config import Config


def _cli_version() -> str:
    """Resolve package version for Click's --version output."""
    try:
        return version("tab2seq")
    except PackageNotFoundError:
        return "0.0.0+unknown"


@click.group(help="tab2seq command line tools")
@click.version_option(version=_cli_version())
def main() -> None:
    """Main tab2seq command group."""


@main.command("init-config")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def init_config(output_path: Path) -> None:
    """Write a default YAML config file."""
    cfg = Config()
    cfg.to_yaml(output_path)
    click.echo(f"Configuration saved to {output_path}")


@main.command("validate-config")
@click.argument("config_path", type=click.Path(path_type=Path))
def validate_config(config_path: Path) -> None:
    """Validate a YAML config file."""
    try:
        _ = Config.from_yaml(config_path)
    except Exception as exc:  # pragma: no cover - behavior validated by tests
        raise click.ClickException(str(exc)) from exc
    click.echo("Configuration is valid")


@main.command("process")
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), required=True)
@click.option("--format", "output_format", type=click.Choice(["csv", "parquet"]), default="parquet")
@click.option("--n-jobs", "n_jobs", type=int, default=1)
def process(
    input_path: Path,
    config_path: Path | None,
    output_dir: Path,
    output_format: str,
    n_jobs: int,
) -> None:
    """Run a minimal processing pipeline stub for tests and demos."""
    if config_path is not None:
        _ = Config.from_yaml(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(
        f"Processing {input_path} -> {output_dir} (format={output_format}, n_jobs={n_jobs})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
