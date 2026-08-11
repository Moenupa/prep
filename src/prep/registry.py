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


def register_loader(
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
        return function

    return decorator


def get_folder_size(path: Path | None) -> str:
    if path is None:
        return ""
    return decimal(
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()),
        precision=0,
    )


def status_table(save_dir: Path) -> Table:
    split_names = get_valid_splits()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset ID", style="cyan", no_wrap=True)
    table.add_column("Data Format", style="green", no_wrap=True)
    table.add_column("Default Source", style="blue")
    table.add_column("Local Path", style="yellow")
    table.add_column("Local Size", style="yellow")
    for split in split_names:
        table.add_column(f"{split:5s}", style="yellow", no_wrap=True)

    pipelines_by_dataset: dict[tuple[str, DataFormat], list[FormatterPipeline]] = {}
    for pipeline in _FORMATTER_REGISTRY.values():
        pipelines_by_dataset.setdefault(
            (pipeline.id_, pipeline.target_format), []
        ).append(pipeline)

    for (_, _), pipelines in sorted(pipelines_by_dataset.items()):
        pipeline = pipelines[0]
        local_path = pipeline.save_dir(save_dir)
        if not local_path.exists() or not any(local_path.glob("*")):
            local_path = None

        default_sources = {pipeline.default_src for pipeline in pipelines}
        default_source = (
            next(iter(default_sources))
            if len(default_sources) == 1
            else ", ".join(sorted(source for source in default_sources if source))
        )
        local_splits = pipeline.find_splits(save_dir)
        table.add_row(
            pipeline.id_,
            pipeline.target_format,
            default_source or "",
            str(local_path or ""),
            get_folder_size(local_path),
            *(
                {True: "✔", False: "✖", None: ""}[local_splits[split]]
                for split in split_names
            ),
            style=None if local_path is not None else "dim",
        )
    return table
