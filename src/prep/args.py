import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from datasets import Dataset

from .constants import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger(__name__)


_DataFormat = Literal["sft", "verl", "eval"]
_Split = Literal["train", "val", "test"]
type DatasetPrepStream = Dataset
type DataFormat = _DataFormat
type Split = _Split
type LoadFn = Callable[[str, Split, LoadArgs], DatasetPrepStream]


def get_valid_splits():
    return get_args(_Split)


def get_valid_formats():
    return get_args(_DataFormat)


@dataclass(frozen=True)
class LoadArgs:
    num_proc: int
    question_cols: list[str]
    question_template: str
    answer_cols: list[str]
    show_first_n: int = 3

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


@dataclass(frozen=True)
class RuntimeArgs:
    # arguments only known at runtime
    override_src: str | None

    save: bool
    save_dir: Path
    save_parquet: bool

    hf: bool
    hf_repo: str
    hf_subset: str | None
    hf_private: bool

    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def do_save(self, d: "Dataset", save_path: Path):
        logger.info(f"{'🌵 DRY\t' if not self.save else '💾\t'} Save -> {save_path!r}")
        if not self.save:
            return

        d.to_parquet(save_path) if self.save_parquet else d.save_to_disk(save_path)

    def do_upload(self, d: "Dataset", split: Split):
        logger.info(
            f"{'🌵 DRY\t' if not self.hf else '☁️\t'} Upload -> {self.hf_repo!r}"
            f" (subset={self.hf_subset!r}, split={split!r}, private={self.hf_private})"
        )
        if not self.hf:
            return

        d.push_to_hub(
            repo_id=self.hf_repo,
            config_name=self.hf_subset or "default",
            split=split,
            private=self.hf_private,
        )
