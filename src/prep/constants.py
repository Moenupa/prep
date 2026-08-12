import logging
import os
from typing import Any

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(__name__)

    logger = logging.getLogger(name)
    return logger


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]


def first_value(entry: dict, keys: list[str]) -> Any | None:
    """Return the value for the first key that exists in entry."""
    for key in keys:
        if entry.get(key) is not None:
            return entry.get(key)
    return None


DEFAULT_QCOLS = ["question", "Question", "problem"]
DEFAULT_ACOLS = [
    "answer",
    "Answer",
    "solution",
    "label",
    "caption",
    "correct_answer",
    "reports",
]
