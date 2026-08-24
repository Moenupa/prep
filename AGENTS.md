# AGENTS.md

This file provides agents with information about the project.

## Commands

Use `make`, `uv`, and `uvx` commands to run code formatting, linting, and tests.

```sh
make format-check  # Check code lint/format (no fix)
make format        # Run linter and formatter
make check         # Run type checker
make test          # Run quick tests
make test-slow     # Run tests with real examples (slow)

# Run a single test file
uv run pytest tests/path/to/test_file.py
```

## Architecture

### Entry Points

- `prep` command -> `src/prep/cli.py:prep()` converts a source dataset into one of the supported target formats.
- `ppls` command -> `src/prep/cli.py:ppls()` lists registered pipelines and local output status under the save root.

### Key Modules

- `src/prep/cli.py`: Typer CLI for conversion and pipeline/status listing.
- `src/prep/api`: core module, modify with care. It includes pipeline definitions, registration, schema, validation and data IO.
- `src/prep/formatter/`: dataset-specific formatter pipeline definitions.
- `src/prep/formatter/auto.py`: generic `vqa` pipelines that map common VQA-style datasets into `sft`, `verl`, and `eval`.
- `src/prep/formatter/geo3k.py`: a minimal example to illustrate how to add a dataset-specific formatter pipeline.

### Runtime Flow

- `prep` resolves a `FormatterPipeline`, loads data through the registered loader, casts to the target schema when possible, validates a sample window, previews examples, then optionally saves to disk.
- Generic pipelines use `ProcArgs` fields such as `question_cols`, `answer_cols`, `option_cols`, and templates to adapt different dataset column layouts.
- Local outputs are organized under `out/<format>/<pipeline_id>/<split>` by default, with parquet output writing to `out/<format>/<pipeline_id>/<split>.parquet`.

## Adding Support for a New Dataset

- Run `uv run ppls --list-format` to inspect supported output formats, then `uv run ppls` to see registered pipeline IDs and locally materialized outputs.
- Prefer the generic `vqa` pipelines first. They can load local files/directories or Hugging Face datasets and can often be adapted just by overriding `--q-cols`, `--a-cols`, `--op-cols`, `--q-template`, or `--a-template`.
- If the generic mapper is insufficient, add a dataset-specific formatter module under `src/prep/formatter/` and register each supported `(id_, format, split)` combination with `@formatter(...)`.
- Only maintain dataset-specific field mapping in the formatter module. Shared feature schemas and validation belong in `src/prep/api/types.py` and `src/prep/api/validate.py`, which are automatically checked against the output format.
- Avoid `/` in formatter IDs; the registry enforces this. 
- Use `org/data@subset` to denote subsets when loading from remote.

## Code Style

- `make format` or `ruff` for formatting with Google-style docstrings
- `make check` or `ty` for static analysis
- Python 3.12+ syntax
