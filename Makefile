.PHONY: help format-check format lint test test-slow examples

check_dirs := src tests

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

all: format lint

help:
	@echo "Available targets:"
	@echo "  format-check:  Check code formatting (no fix)"
	@echo "  format:        Format code"
	@echo "  lint:          Run lint checks"
	@echo "  test:          Run tests"
	@echo "  examples:      Run examples"

format-check:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

format:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

lint:
	$(TOOL) ty check $(check_dirs)

test:
	$(RUN) pytest tests -n 8

test-slow:
	RUN_SLOW=1 $(RUN) pytest tests -n 8 -rA

examples:
	$(MAKE) -C examples
