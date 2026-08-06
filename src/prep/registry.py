from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple, get_args

from datasets import Dataset
from rich.table import Table

from .constants import get_logger
from .validator import sft_mm_features, validate_openai_messages, verl_mm_features

logger = get_logger(__name__)

DataFormat = Literal["sft", "verl", "eval"]
Split = Literal["train", "val", "test"]

type DataFormat_ = DataFormat
type Split_ = Split
type LoadFn = Callable[[str, Split_], Dataset]


class DatasetRegistryItem(NamedTuple):
    load_fn: LoadFn
    default_src: str | None


_DATASET_REGISTRY: dict[tuple[str, DataFormat_, Split_], DatasetRegistryItem] = {}


def register_loader(
    data_id: str,
    data_format: DataFormat_,
    split: Split_,
    default_src: str | None = None,
) -> Callable[[LoadFn], LoadFn]:
    def decorator(function: LoadFn) -> LoadFn:
        if (data_id, data_format, split) in _DATASET_REGISTRY:
            raise ValueError(
                "Preprocessor already registered for dataset"
                f" id={data_id!r}, format={data_format!r}, split={split!r}"
            )

        _DATASET_REGISTRY[(data_id, data_format, split)] = DatasetRegistryItem(
            function,
            default_src,
        )
        return function

    return decorator


def load(
    data_id: str,
    fmt: DataFormat_,
    split: Split_,
    override_src: str | None,
    image_tag: str = "<image>",
) -> Dataset:
    if (data_id, fmt, split) not in _DATASET_REGISTRY:
        raise KeyError(
            f"Undefined pipeline for id={data_id!r}, format={fmt!r}, split={split!r}."
        )

    loader = _DATASET_REGISTRY[(data_id, fmt, split)]
    path = override_src or loader.default_src
    if path is None:
        raise ValueError(
            f"Dataset {data_id!r} has no local/remote source to load from."
            " Pass --src to override or register a default source in loading function."
        )
    d = loader.load_fn(path, split)
    sample = d[0]
    try:
        match fmt:
            case "verl":
                d = d.cast(verl_mm_features)
                validate_openai_messages(
                    sample["prompt"],
                    expected_n_img=len(sample.get("images", [])),
                    img_tag=image_tag,
                )
            case "sft":
                d = d.cast(sft_mm_features)
                validate_openai_messages(
                    sample["messages"],
                    expected_n_img=len(sample.get("images", [])),
                    img_tag=image_tag,
                )
            case "eval":
                n_tags = sample["question"].count(image_tag)
                n_img = len(sample.get("images", []))
                assert n_tags == n_img, (
                    f"Mismatch: number of images {n_img} != {n_tags} {image_tag} tags."
                )
    except Exception as e:
        logger.error(
            f"⚠️\tValidation failed {data_id!r}, format={fmt!r}, split={split!r}\n⚠️ {e}"
        )

    return d


class DataInfo(NamedTuple):
    id: str
    fmt: DataFormat_
    default_src: str | None
    save_dir: Path

    def save_to(self, parquet: bool, split: Split_) -> Path:
        save_path = self.save_dir / self.fmt / self.id / split
        if parquet:
            save_path = save_path.with_suffix(".parquet")

        return save_path

    @property
    def local_path(self) -> Path | None:
        save_dir = self.save_dir / self.fmt / self.id
        if save_dir.exists() and any(save_dir.glob("*")):
            return save_dir

        return None

    @property
    def local_splits(self) -> dict[Split_, bool | None]:
        splits = {}
        for split in ["train", "val", "test"]:
            # not even registered, then None
            if (self.id, self.fmt, split) not in _DATASET_REGISTRY:
                splits[split] = None
                continue

            splits[split] = self.local_path is not None and any(
                self.local_path.glob(f"{split}*")
            )
        return splits


def get_data_info(save_dir: Path) -> list[DataInfo]:
    return [
        DataInfo(name, fmt, src, save_dir=save_dir)
        for name, fmt, src in {
            (name, fmt, item.default_src)
            for (name, fmt, _), item in _DATASET_REGISTRY.items()
        }
    ]


def status_table(save_dir: Path) -> Table:
    split_names = get_args(Split)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset ID", style="cyan", no_wrap=True)
    table.add_column("Data Format", style="green", no_wrap=True)
    table.add_column("Default Source", style="blue")
    table.add_column("Local Path", style="yellow")
    for split in split_names:
        table.add_column(f"{split:5s}", style="yellow", no_wrap=True)

    for row in sorted(get_data_info(save_dir), key=lambda x: (x.id, x.fmt)):
        table.add_row(
            row.id,
            row.fmt,
            row.default_src or "",
            str(row.local_path or ""),
            *(
                {True: "✔", False: "✖", None: ""}[row.local_splits.get(split, None)]
                for split in split_names
            ),
            style=None if row.local_path is not None else "dim",
        )
    return table


def save_to(
    d: Dataset,
    data_id: str,
    fmt: DataFormat_,
    split: Split_,
    dry_run: bool,
    save_dir: Path,
    parquet: bool,
):
    save_path = DataInfo(data_id, fmt, None, save_dir=save_dir).save_to(
        parquet=parquet, split=split
    )
    logger.info(f"{'🌵' if dry_run else '💾'}\tSaving {data_id!r} -> {save_path!r}")
    if dry_run:
        return
    d.to_parquet(save_path) if parquet else d.save_to_disk(save_path)


def peek(d: Dataset, first_n: int):
    logger.info(d)
    for i in range(min(first_n, len(d))):
        logger.info(d[i])
