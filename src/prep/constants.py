import logging
import os
import re

from rich.logging import RichHandler


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]


DEFAULT_QCOLS = ["question", "Question", "problem"]
DEFAULT_QTEMP = "{im_tags}{question}{options}"
DEFAULT_ACOLS = [
    "answer",
    "Answer",
    "solution",
    "label",
    "caption",
    "correct_answer",
    "reports",
]
DEFAULT_ATEMP = "{answer}"
DEFAULT_OPCOLS = ["options", "choices"] + [f"choice_{op}" for op in "abcdefghij"]
IMAGE_TAG = "<image>"
FORMATTING_PATTERN = re.compile(r"ANSWER:|<answer>|</answer>|\\boxed\{")
ID_PATTERN = re.compile(r"^[a-zA-Z0-9\._-]+$")

SAVE_PREFIX = "💾\t"
HF_PREFIX = "☁️\t"
SKIP = "SKIP"

WARN_PREFIX = "⚠️\t"
ERROR_PREFIX = "⛔\t"

logging.basicConfig(
    format="%(message)s",
    level=os.getenv("LOG_LEVEL") or logging.INFO,
    handlers=[RichHandler(show_time=False, show_path=False, rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.ERROR)
