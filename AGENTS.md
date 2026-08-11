# AGENTS.md

This file provides agents with information about the project.

## Commands

Use `make`, `uv`, `uvx` commands to run code style checks, quality checks, and tests.

```sh
make format-check  # Check code formatting (no fix)
make format        # Format code
make lint          # Run lint checks
make test          # Run tests
make examples      # Run examples

# Run a single test file
uv run pytest tests/path/to/test_file.py
```

## Architecture

### Entry Points

- `prep` command -> `src/prep/cli.py:prep()` runs dataset conversion.
- `ppls` command -> `src/prep/cli.py:status()` renders the registered dataset/status table for the local cache directory.

### Key Modules

- `src/prep/args.py`: typed CLI/runtime argument containers and validation.
- `src/prep/registry.py`: formatter registration, pipeline lookup, save-path helpers, and status table generation.
- `src/prep/validator.py`: target data schemas (features) and sample validation.
- `src/prep/formatter/`: dataset-specific formatter pipeline definitions.
- `src/prep/formatter/auto.py`: generic VQA-to-SFT/VERL converters for datasets that follow common conventions.

## Adding Support for a New Dataset

- Run `uv run ppls` to see if it is already supported, then check whether `src/prep/formatter/auto.py` can convert it with the generic loaders.
- If the generic loader is not enough, add a dataset-specific formatter module under `src/prep/formatter/` and register loaders with `register_loader(...)` for each supported target format/split.
- Prefer keeping dataset-specific field mapping inside the formatter module and shared validation/feature logic inside `prep.registry` or `prep.validator`.

## Code Style

- Ruff for linting and formatting (Google-style docstrings)
- Python 3.12+ syntax
