import os
import re
from typing import Any


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]


def first_value(entry: dict, keys: list[str]) -> Any | None:
    """Return the value for the first key that exists in entry."""
    for key in keys:
        if entry.get(key) is not None:
            return entry.get(key)
    return None


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

SAVE_PREFIX = "💾\t"
HF_PREFIX = "☁️\t"
SKIP = "SKIP"

WARN_PREFIX = "⚠️\t"
ERROR_PREFIX = "⛔\t"
