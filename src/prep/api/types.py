import logging
from dataclasses import dataclass, field
from typing import Literal, get_args

from datasets import Dataset, Features, Image, List, Value

from ..constants import (
    DEFAULT_ACOLS,
    DEFAULT_ATEMP,
    DEFAULT_OPCOLS,
    DEFAULT_QCOLS,
    DEFAULT_QTEMP,
)
from .log import get_logger

_DataFormat = Literal["sft", "verl", "eval", "show"]
_Split = Literal["train", "val", "test"]
type DataFormat = _DataFormat
type Split = _Split
logger = get_logger()


def get_valid_splits() -> list[Split]:
    return list(get_args(_Split))


def get_valid_formats() -> list[DataFormat]:
    return list(filter(lambda x: x != "show", get_args(_DataFormat)))


VERL_FEAT = Features(
    images=List(Image(decode=True)),
    data_source=Value("string"),
    prompt=List(
        {
            "role": Value("string"),
            "content": Value("large_string"),
        }
    ),
    ability=Value("string"),
    reward_model={
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    extra_info={
        "split": Value("string"),
        "index": Value("string"),
        # feedback, CoT, or hint to guide better answers
        "explanation": Value("large_string"),
        # any miscellaneous info accepting json.dumps() stuff
        # this is for compatiblity with multiple datasets, supporting any structure
        "misc": Value("large_string"),
    },
)

SFT_FEAT = Features(
    images=List(Image(decode=True)),
    messages=List(
        {
            "role": Value("string"),
            "content": Value("large_string"),
        }
    ),
    id=Value("string"),
    extra_info=Value("large_string"),
)

EVAL_FEAT = Features(
    id=Value("string"),
    images=List(Image(decode=True)),
    question=Value("string"),
    options=List(Value("string")),
    answer=Value("string"),
)


@dataclass(frozen=True)
class ProcArgs:
    """A lite wrapper for arguments for formatter pipelines.

    Raises:
        ValueError: If ``num_proc`` is not positive.
        ValueError: If ``question_cols`` is empty.
        ValueError: If ``answer_cols`` is empty.
        ValueError: If ``question_template`` is empty.
    """

    num_proc: int

    question_cols: list[str] = field(default_factory=lambda: DEFAULT_QCOLS)
    question_template: str = DEFAULT_QTEMP

    option_cols: list[str] = field(default_factory=lambda: DEFAULT_OPCOLS)

    answer_cols: list[str] = field(default_factory=lambda: DEFAULT_ACOLS)
    answer_template: str = DEFAULT_ATEMP

    # to fill in verl fields, this does not affect verl training
    verl_ability: str = "math"
    verl_style: str = "rule"

    show_first_n: int = 3
    max_samples: int | None = None

    def __post_init__(self):
        if self.num_proc <= 0:
            raise ValueError(f"Invalid num_proc: {self.num_proc}")
        if not self.question_cols:
            raise ValueError(f"Invalid question_cols: {self.question_cols}")
        if not self.answer_cols:
            raise ValueError(f"Invalid answer_cols: {self.answer_cols}")
        if not self.question_template:
            raise ValueError(f"Invalid question_template: {self.question_template}")
        # we explicitly allow no '{question}' in template for captioning datasets
        if "{question}" not in self.question_template:
            logger.warning(
                "question_template does not contain '{question}' placeholder."
            )

    def peek(self, d: "Dataset", level: int = logging.INFO):
        logger.log(level, d)

        for i in range(min(self.show_first_n, len(d))):
            logger.log(level, d[i])


class RegistrationError(Exception):
    """Formatter pipeline registration error."""
