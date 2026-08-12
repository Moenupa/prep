from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
)

_FORMATS = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".arrow": "arrow",
    ".txt": "text",
    ".text": "text",
}


def resolve_split(split: str, available: list[str]) -> str:
    # direct hit is preferred
    if split in available:
        return split

    # translate if "val" -> "validation" if exists
    # we want to be generous here to facilitate IO
    if split == "val" and "validation" in available:
        return "validation"

    raise ValueError(f"Split {split!r} not found in dataset: {available}")


def load_local(source: str, split: str) -> Dataset | None:
    try:
        d = load_from_disk(source)
    except Exception:
        return None

    if isinstance(d, Dataset):
        return d

    if not isinstance(d, DatasetDict):
        raise RuntimeError(f"Unexpected dataset type: {type(d)} {d}")

    return d[resolve_split(split, list(map(str, d.keys())))]


def adaptive_load_dataset(
    source: str, split: str, nproc: int | None = None, subset: str | None = None
) -> Dataset:
    path = Path(source)

    d = None
    # non-local -> HF remote
    if not path.exists():
        return load_dataset(
            source,
            subset=subset,
            split=resolve_split(split, get_dataset_split_names(source, subset=subset)),
            num_proc=nproc,
        )

    # local -> 1. dir 2. file
    elif path.is_dir():
        d = load_local(source, split) or load_dataset(
            source, subset=subset, split=split, num_proc=nproc
        )
        return d
    elif path.is_file():
        if path.suffix not in _FORMATS:
            raise ValueError(f"Unsupported file format: {path.suffix!r}")

        d = load_dataset(_FORMATS[path.suffix], data_files=[source], split="train")
        return d

    if d is None:
        raise ValueError(f"Invalid source path: {source!r}")
