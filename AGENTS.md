# AGENTS.md

This file provides agents with information about the project.

## Commands

Use `make`, `uv`, and `uvx` commands to run code formatting, linting, and tests.

```sh
# lint, format, and type check
make
# Check code lint/format (no fix)
make format-check
# Run linter and formatter
make format
# Run type checker
make check
# Run quick tests
make test
# Run tests with real examples (slow)
make test-slow
# Run a single test file
uv run pytest tests/path/to/test_file.py
```

## Architecture

### Entry Points

- `prep` command -> `src/prep/cli.py:prep()` converts a source dataset into one of the supported target formats.
- `ppls` command -> `src/prep/cli.py:ppls()` lists registered pipelines and local output status under the save root.

### Key Modules

- `src/prep/cli.py`: Typer CLI for conversion and pipeline/status listing.
- `src/prep/formatter/`: dataset-specific formatter pipeline definitions.
- `src/prep/formatter/auto_*.py`: generic `auto` pipelines.
- `src/prep/formatter/geo3k.py`: a minimal example to illustrate how to add a dataset-specific formatter pipeline.

### Runtime Flow

- `prep` resolves a `FormatterPipeline`, loads data through the registered loader, casts to the target schema when possible, validates a sample window, previews examples, then optionally saves to disk.
- `ProcArgs` keeps process-time arguments such as column names and templates to adapt different dataset column layouts.
- Local outputs are organized under `out/<format>/<pipeline_id>/<split>` or `out/<format>/<pipeline_id>/<split>.parquet`.

## Adding Support for a New Dataset

- Run `uv run ppls --list-format` to inspect supported output formats, then `uv run ppls` to see registered pipeline IDs and locally materialized outputs.
- Prefer the generic `auto` pipelines first. They can load local files/directories or Hugging Face datasets and can often be adapted just by overriding `--q-cols`, `--a-cols`, `--op-cols`, `--q-template`, or `--a-template`.
- If the generic mapper is insufficient, add a dataset-specific formatter module under `src/prep/formatter/` and register each supported `(id_, format, split)` combination with `@formatter(...)`.
- Only maintain dataset-specific field mapping in the formatter module. Shared feature schemas and validation belong in `src/prep/api/types.py` and `src/prep/api/validate.py`, which are automatically checked against the output format.
- Avoid `/` in formatter IDs; the registry enforces this. 
- Use `org/data@subset` to denote subsets when loading from remote.

## Code Style

Python 3.12 with Google-style docstrings
