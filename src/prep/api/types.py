import logging
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from ..constants import (
    DEFAULT_ACOLS,
    DEFAULT_ATEMP,
    DEFAULT_OPCOLS,
    DEFAULT_QCOLS,
    DEFAULT_QTEMP,
)
from .log import get_logger
from .transform import validate_transform_names

_DataFormat = Literal["sft", "verl", "eval", "clip", "cls", "show"]
_Split = Literal["train", "val", "test"]
type DataFormat = _DataFormat
type Split = _Split
logger = get_logger()


def get_valid_splits() -> list[Split]:
    return list(get_args(_Split))


def get_valid_formats() -> list[DataFormat]:
    return list(filter(lambda x: x != "show", get_args(_DataFormat)))


@dataclass(frozen=True)
class ProcArgs:
    """A lite wrapper for arguments for formatter pipelines.

    Raises:
        ValueError: If ``num_proc`` is not positive.
        ValueError: If ``question_cols`` is empty.
        ValueError: If ``answer_cols`` is empty.
        ValueError: If ``question_template`` is empty.
        KeyError: If ``transforms`` contains an unknown transform name.
    """

    num_proc: int = 1

    question_cols: list[str] = field(default_factory=lambda: DEFAULT_QCOLS)
    question_template: str = DEFAULT_QTEMP

    option_cols: list[str] = field(default_factory=lambda: DEFAULT_OPCOLS)

    answer_cols: list[str] = field(default_factory=lambda: DEFAULT_ACOLS)
    answer_template: str = DEFAULT_ATEMP
    labels: list[str] = field(default_factory=lambda: [])

    # image-to-image transforms applied in order during conversion
    transforms: list[str] = field(default_factory=lambda: [])

    # to fill in verl fields, this does not affect verl training
    verl_ability: str = "math"
    verl_style: str = "rule"

    show_first_n: int = 3
    show_last_n: int = 0
    max_samples: int | None = None

    # None for no shuffling, negative for random seed, non-negative for fixed seed
    seed: int | None = None

    def __post_init__(self):
        if self.num_proc <= 0:
            raise ValueError(f"Invalid num_proc: {self.num_proc}")
        if not self.question_cols:
            raise ValueError(f"Invalid question_cols: {self.question_cols}")
        if not self.answer_cols:
            raise ValueError(f"Invalid answer_cols: {self.answer_cols}")
        if not self.question_template:
            raise ValueError(f"Invalid question_template: {self.question_template}")
        validate_transform_names(self.transforms)
        # we explicitly allow no '{question}' in template for captioning datasets
        if "{question}" not in self.question_template:
            logger.warning(
                "question_template does not contain '{question}' placeholder."
            )

    def peek(self, d: Any, level: int = logging.INFO):
        logger.log(level, d)

        for i in range(min(self.show_first_n, len(d))):
            logger.log(level, d[i])

        for i in range(min(self.show_last_n, len(d))):
            logger.log(level, d[-(i + 1)])


class RegistrationError(Exception):
    """Formatter pipeline registration error."""
