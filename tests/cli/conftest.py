"""Shared fixtures for end-to-end CLI tests.

Each test invokes the Typer app in-process via ``CliRunner`` and compares the
captured stdout (never stderr) against an exact snapshot. All env vars bound
to CLI options are cleared, and ``COLUMNS``/``NO_COLOR`` are pinned so that
rich renders logs at a fixed width without ANSI codes.
"""

import json
from fnmatch import fnmatch

import pytest
from typer.testing import CliRunner

# env vars bound to typer options in prep.cli; keep in sync with cli.py
CLI_ENVVARS = (
    "SRC",
    "HEAD",
    "TAIL",
    "NPROC",
    "SEED",
    "SAVE",
    "SAVE_DIR",
    "SAVE_PARQ",
    "SAVE_NPROC",
    "SAVE_PREVIEW",
    "HF",
    "HF_REPO",
    "HF_SUBSET",
    "HF_PRIVATE",
    "HF_NPROC",
    "MAX_SAMPLES",
    "Q_COLS",
    "Q_TEMP",
    "OP_COLS",
    "A_COLS",
    "A_TEMP",
    "VERL_ABILITY",
    "VERL_STYLE",
    "LABELS",
    "TRANSFORMS",
    "UI",
    "COLUMNS",
    "NO_COLOR",
    "FORCE_COLOR",
)


@pytest.fixture(autouse=True)
def cli_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Isolate the CLI from the host environment and fix rich rendering."""
    import logging

    from rich.logging import RichHandler

    for var in CLI_ENVVARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("NO_COLOR", "1")
    # RichHandler captures an os.environ snapshot at import time, while pytest
    # capture swaps os.environ per test; pin the cached console size directly
    # so log records always render at a fixed width.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            console = handler.console
            prev_size = (console._width, console._height)
            console.size = (120, 25)

            def restore(prev=prev_size, c=console):
                c._width, c._height = prev

            request.addfinalizer(restore)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner for invoking the prep/ppls commands in-process."""
    return CliRunner()


@pytest.fixture
def qa_jsonl(tmp_path) -> str:
    """A 10-row question/answer source in jsonl format."""
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"question": f"What is {i}+{i}?", "answer": str(2 * i)})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    return path.name


@pytest.fixture
def mc_jsonl(tmp_path) -> str:
    """A 10-row multiple-choice source in jsonl format."""
    path = tmp_path / "mc.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question": f"Pick the even number among the choices (set {i}).",
                    "options": [str(2 * i + 1), str(2 * i + 2), str(2 * i + 3)],
                    "answer": "B",
                }
            )
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    return path.name


def _rstrip_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


@pytest.fixture
def assert_stdout():
    """Assert exact stdout equality (modulo rich's right-padding of log lines)."""

    def _assert(result, expected: str, exit_code: int = 0) -> None:
        assert result.exit_code == exit_code
        for line, exp in zip(
            _rstrip_lines(result.stdout), _rstrip_lines(expected), strict=True
        ):
            if "*" in exp:
                assert fnmatch(line, exp), (
                    f"line '{line}' does not match pattern '{exp}'"
                )
            else:
                assert line == exp

    return _assert
