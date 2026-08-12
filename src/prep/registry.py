from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.filesize import decimal
from rich.table import Table

from .args import (
    DataFormat,
    DatasetPrepStream,
    LoadArgs,
    LoadFn,
    Split,
    get_valid_formats,
    get_valid_splits,
)
from .constants import get_logger
from .validator import sft_mm_features, validate_openai_messages, verl_mm_features

logger = get_logger(__name__)


_FORMATTER_REGISTRY: dict[tuple[str, DataFormat, Split], "FormatterPipeline"] = {}


@dataclass(frozen=True)
class FormatterArgs:
    id_: str
    target_format: DataFormat
    split: Split

    def __post_init__(self):
        if (self.id_, self.target_format, self.split) not in _FORMATTER_REGISTRY:
            raise ValueError(f"Undefined pipeline for {self}.")

    def __str__(self) -> str:
        return f"Formatter(id={self.id_!r}, target_format={self.target_format!r}, split={self.split!r})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def pipeline(self) -> "FormatterPipeline":
        return _FORMATTER_REGISTRY[(self.id_, self.target_format, self.split)]

    def save_dir(self, save_dir: Path) -> Path:
        return save_dir / self.target_format / self.id_

    def save_path(
        self, save_dir: Path, parquet: bool, split: Split | None = None
    ) -> Path:
        save_path = self.save_dir(save_dir) / (split or self.split)
        if parquet:
            save_path = save_path.with_suffix(".parquet")
        return save_path

    def find_splits(self, save_dir: Path) -> dict:
        splits = {}
        for split in get_valid_splits():
            # not even registered, then None
            if (self.id_, self.target_format, split) not in _FORMATTER_REGISTRY:
                splits[split] = None
                continue

            # either parquet exists or folder unempty
            splits[split] = (
                any(self.save_path(save_dir, parquet=False, split=split).glob("*"))
                or self.save_path(save_dir, parquet=True, split=split).exists()
            )
        return splits


@dataclass(frozen=True)
class FormatterPipeline(FormatterArgs):
    load_fn: LoadFn
    default_src: str | None

    def __post_init__(self):
        # deliberate skipping check-exist, because this applies to registration
        if (self.id_, self.target_format, self.split) in _FORMATTER_REGISTRY:
            raise ValueError(f"Pipeline already registered for {self}.")

    def __str__(self) -> str:
        return f"Formatter(id={self.id_!r}, target_format={self.target_format!r}, split={self.split!r}, default_src={self.default_src!r})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def pipeline(self) -> "FormatterPipeline":
        return self

    def check_sample(self, sample: dict, image_tag: str) -> None:
        match self.target_format:
            case "verl":
                validate_openai_messages(
                    sample["prompt"],
                    expected_n_img=len(sample.get("images", [])),
                    img_tag=image_tag,
                )
            case "sft":
                validate_openai_messages(
                    sample["messages"],
                    expected_n_img=len(sample.get("images", [])),
                    img_tag=image_tag,
                )
            case "eval":
                if not isinstance(sample.get("question"), str):
                    return

                n_tags = sample["question"].count(image_tag)
                n_img = len(sample.get("images", []))
                assert n_tags == n_img, (
                    f"Mismatch: number of images {n_img} != {n_tags} {image_tag} tags."
                )

    def load(self, override_src: str | None, loadargs: LoadArgs) -> DatasetPrepStream:
        path = override_src or self.default_src
        if path is None:
            raise ValueError(
                f"Dataset {self.id_!r} has no local/remote source to load from."
                " Pass --src to override or register a default source in loading function."
            )
        d = self.load_fn(path, self.split, loadargs)
        try:
            match self.target_format:
                case "verl":
                    d = d.cast(verl_mm_features)
                case "sft":
                    d = d.cast(sft_mm_features)
        except Exception as e:
            logger.error(f"⚠️\tCasting failed {str(self)}\n⚠️\t{e}")
        try:
            self.check_sample(d[0], image_tag="<image>")
        except Exception as e:
            logger.error(f"⚠️\tValidation failed {str(self)}\n⚠️\t{e}")

        loadargs.peek(d)
        return d


def formatter(
    id_: str,
    target_format: DataFormat,
    split: Split,
    default_src: str | None = None,
) -> Callable[[LoadFn], LoadFn]:
    def decorator(function: LoadFn) -> LoadFn:
        assert "/" not in id_, "pipeline ID should not contain '/'"

        _FORMATTER_REGISTRY[(id_, target_format, split)] = FormatterPipeline(
            id_=id_,
            target_format=target_format,
            split=split,
            load_fn=function,
            default_src=default_src,
        )
        logger.debug(
            f"🗂\tRegistered {id_!r} (fmt={target_format!r}, split={split!r}, src={default_src!r})"
        )
        return function

    return decorator


def get_folder_size(path: Path | None) -> str:
    if path is None:
        return ""
    return decimal(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
        precision=0,
    )


def status_table(save_dir: Path):
    pipelines_by_dataset: dict[tuple[str, DataFormat], FormatterPipeline] = {}
    for pipeline in _FORMATTER_REGISTRY.values():
        pipelines_by_dataset[(pipeline.id_, pipeline.target_format)] = pipeline
    split_names = get_valid_splits()

    for each_format in get_valid_formats():
        table = Table(show_header=True, title=each_format, header_style="bold magenta")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Default Source", style="green")
        table.add_column("Local Path", style="blue")
        table.add_column("Size", style="yellow")
        for split in split_names:
            table.add_column(split, style="yellow", no_wrap=True)

        for (_, target_format_), pipeline in sorted(pipelines_by_dataset.items()):
            if target_format_ != each_format:
                continue
            local_dir = pipeline.save_dir(save_dir)
            if not local_dir.exists() or not any(local_dir.glob("*")):
                local_dir = None

            local_splits = pipeline.find_splits(save_dir)
            table.add_row(
                pipeline.id_,
                pipeline.default_src or "",
                str(local_dir or ""),
                get_folder_size(local_dir),
                *(
                    {True: "✔", False: "✖", None: ""}[local_splits[split]]
                    for split in split_names
                ),
                style=None if local_dir is not None else "dim",
            )
        yield table
