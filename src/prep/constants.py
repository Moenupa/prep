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

NUM_PROC = int(os.getenv("NUM_PROC", "16"))
QUESTION_COLS = os.getenv("Q_COLS", "question,Question,problem,Problem").split(",")
QUESTION_TEMPLATE = os.getenv("Q_TEMPLATE", "{question}").strip()
ANSWER_COLS = os.getenv("A_COLS", "answer,Answer,solution,label").split(",")


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
        if val := entry.get(key):
            return val
    return None
