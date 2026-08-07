from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from datasets import Dataset

from .constants import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger(__name__)


_DataFormat = Literal["sft", "verl", "eval"]
_Split = Literal["train", "val", "test"]
type DataFormat = _DataFormat
type Split = _Split
type LoadFn = Callable[[str, Split, LoadArgs], Dataset]


def get_valid_splits():
    return get_args(_Split)


def get_valid_formats():
    return get_args(_DataFormat)


@dataclass(frozen=True)
class LoadArgs:
    num_proc: int
    question_cols: list[str] = field(
        default_factory=lambda: ["question", "Question", "problem", "Problem"]
    )
    question_template: str = "{question}"
    answer_cols: list[str] = field(
        default_factory=lambda: ["answer", "Answer", "solution", "label"]
    )

    def __post_init__(self):
        if self.num_proc <= 0:
            raise ValueError(f"Invalid num_proc: {self.num_proc}")
        if not self.question_cols:
            raise ValueError(f"Invalid question_cols: {self.question_cols}")
        if not self.answer_cols:
            raise ValueError(f"Invalid answer_cols: {self.answer_cols}")
        if not self.question_template or "{question}" not in self.question_template:
            raise ValueError(
                f"Invalid question_template: {self.question_template}"
                " Must contain '{question}' placeholder."
            )


@dataclass(frozen=True)
class RuntimeArgs:
    # arguments only known at runtime
    override_src: str | None
    show_first_n: int

    save_dir: Path
    save_parquet: bool

    hf: bool
    hf_repo: str
    hf_subset: str | None
    hf_private: bool

    dry_run: bool

    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def peek(self, d: "Dataset"):
        logger.info(d)

        for i in range(min(self.show_first_n, len(d))):
            logger.info(d[i])

    def save(self, d: "Dataset", save_path: Path):
        logger.info(f"{'🌵' if self.dry_run else '💾'}\tSaving to {save_path!r}")
        if self.dry_run:
            return

        d.to_parquet(save_path) if self.save_parquet else d.save_to_disk(save_path)

    def upload(self, d: "Dataset", split: Split):
        if self.hf_subset is None:
            return

        logger.info(
            f"☁️\tUploading to {self.hf_repo!r}"
            f" (subset={self.hf_subset!r}, split={split!r}, private={self.hf_private})"
        )
        if self.dry_run:
            return

        d.push_to_hub(
            repo_id=self.hf_repo,
            config_name=self.hf_subset,
            split=split,
            private=self.hf_private,
        )
