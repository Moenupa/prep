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
LoadFn = Callable[[str, Split], Dataset]

type DataFormat_ = DataFormat
type Split_ = Split
type LoadFn_ = LoadFn


class DatasetRegistryItem(NamedTuple):
    load_fn: LoadFn_
    remote_path: str | None


_DATASET_REGISTRY: dict[tuple[str, DataFormat_, Split_], DatasetRegistryItem] = {}


def register_loader(
    data_id: str,
    data_format: DataFormat_,
    split: Split_,
    hf_path: str | None = None,
) -> Callable[[LoadFn_], LoadFn_]:
    def decorator(function: LoadFn_) -> LoadFn_:
        if (data_id, data_format, split) in _DATASET_REGISTRY:
            msg = f"Preprocessing function already registered for data_id={data_id!r}, data_format={data_format!r}, and split={split!r}"
            raise ValueError(msg)

        _DATASET_REGISTRY[(data_id, data_format, split)] = DatasetRegistryItem(
            function,
            hf_path,
        )
        return function

    return decorator


def load(
    data_id: str, fmt: DataFormat_, split: Split_, local_path: str | None
) -> Dataset:
    if (data_id, fmt, split) not in _DATASET_REGISTRY:
        msg = f"❓ Undefined pipeline for id={data_id!r}, format={fmt!r}, split={split!r}."
        raise KeyError(msg)

    loader = _DATASET_REGISTRY[(data_id, fmt, split)]
    path = local_path or loader.remote_path
    if path is None:
        raise ValueError(
            f"🚨 Dataset {data_id!r} has no local path or remote path to load from."
        )
    d = loader.load_fn(path, split)
    sample = d[0]
    try:
        match fmt:
            case "verl":
                d = d.cast(verl_mm_features)
                validate_openai_messages(
                    sample["prompt"], expected_n_img=len(sample.get("images", []))
                )
            case "sft":
                d = d.cast(sft_mm_features)
                validate_openai_messages(
                    sample["messages"], expected_n_img=len(sample.get("images", []))
                )
            case "eval":
                assert sample["question"].count("<image>") == len(
                    sample.get("images", [])
                )
    except Exception as e:
        logger.error(
            f"⚠️ Validation failed {data_id!r}, format={fmt!r}, split={split!r}\n⚠️ {e}"
        )

    return d


class DataInfo(NamedTuple):
    id: str
    fmt: DataFormat_
    remote_path: str | None
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
        if self.local_path is None:
            return {}

        splits = {}
        for split in ["train", "val", "test"]:
            # not even registered, then None
            if (self.id, self.fmt, split) not in _DATASET_REGISTRY:
                splits[split] = None
                continue

            splits[split] = any(self.local_path.glob(f"{split}*"))
        return splits


def get_data_info(save_dir: Path) -> list[DataInfo]:
    return [
        DataInfo(name, fmt, remote_path, save_dir=save_dir)
        for name, fmt, remote_path in {
            (name, fmt, item.remote_path)
            for (name, fmt, _), item in _DATASET_REGISTRY.items()
        }
    ]


def status_table(save_dir: Path) -> Table:
    split_names = get_args(Split)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset ID", style="cyan", no_wrap=True)
    table.add_column("Data Format", style="green", no_wrap=True)
    table.add_column("Remote Path", style="blue")
    table.add_column("Local Path", style="yellow")
    for split in split_names:
        table.add_column(f"{split:5s}", style="yellow", no_wrap=True)

    for row in sorted(get_data_info(save_dir), key=lambda x: (x.fmt, x.id)):
        table.add_row(
            row.id,
            row.fmt,
            row.remote_path or "",
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
    logger.info(f"{'🌵' if dry_run else '💾'} Saving {data_id!r} -> {save_path!r}")
    if dry_run:
        return
    d.to_parquet(save_path) if parquet else d.save_to_disk(save_path)


def peek(d: Dataset, first_n: int):
    logger.info(d)
    for i in range(min(first_n, len(d))):
        logger.info(d[i])
