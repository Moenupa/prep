# AGENTS.md

## Commands

Use `make`, `uv`, and `uvx` commands to run code formatting, linting, and tests.

```sh
make                # lint, format, and type check
make format-check   # Check code lint/format (no fix)
make format         # lint and format
make check          # type check
make test           # run test (lite)
make test-slow      # run test (full)
uv run pytest /path/to/test_file.py # Run specific test files
```

## Architecture

- `prep` -> `src/prep/cli.py:prep()` --- convert dataset to target format.
- `ppls` -> `src/prep/cli.py:ppls()` --- list pipelines and local output status.
- `src/prep/formatter/` --- dataset-specific pipeline definitions; `auto_*.py` for generic pipelines.
- Runtime: resolve `FormatterPipeline` -> load data and format -> cast schema -> validate -> preview -> save.
- `ProcArgs` holds column names and templates. Outputs go to `out/<format>/<pipeline_id>/<split>`.

## Adding a New Dataset

1. `uv run ppls --list-format` to see formats, `uv run ppls` to see registered pipelines.
2. Try generic `auto` pipelines first — adapt via auto conversion options listed in `prep --help`.
3. If insufficient, add a module under `src/prep/formatter/` and register with `@formatter(...)`.
4. Formatter should load data, convert fields and return a `datasets.Dataset`.

## Code Style

Python 3.12, Google-style docstrings.
