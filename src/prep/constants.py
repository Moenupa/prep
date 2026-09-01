import os
import re


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
DEFAULT_UI = None if is_env_enabled("UI", "0") else False
# interactive mode off -> disable progress bars
if DEFAULT_UI is not None:
    from datasets import disable_progress_bars

    disable_progress_bars()

SAVE_PREFIX = "💾\t"
HF_PREFIX = "☁️\t"
PREVIEW_PREFIX = "🖼️\t"
SKIP = "X"

WARN_PREFIX = "⚠️\t"
ERROR_PREFIX = "⛔\t"
