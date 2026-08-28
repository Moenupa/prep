import re
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import (
    Dataset,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
)

from .log import get_logger

if TYPE_CHECKING:
    from .types import ProcArgs

logger = get_logger(__name__)

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


def resolve_remote(source: str) -> tuple[str, str | None, str | None]:
    """Resolve a remote dataset source into its components.

    Args:
        source (str): dataset ID in the format of `org/data[@subset][:split]`.

    Returns:
        tuple[str, str | None, str | None]: A tuple containing the source, subset, and split.
    """
    match = re.match(
        r"^(?P<source>[^@]+)(?:@(?P<subset>[^:]+))?(?::(?P<split>.+))?$", source
    )
    if not match:
        raise ValueError(f"Invalid source format: {source!r}")

    return (
        match.group("source"),
        match.group("subset"),
        match.group("split"),
    )


def resolve_split(split: str | None, available: list[str]) -> str:
    if split is None:
        if len(available) == 1:
            return available[0]
        raise ValueError(
            f"Split must be specified for dataset with multiple splits: {available}"
        )

    # direct hit is preferred
    if split in available:
        return split

    # translate if "val" -> "validation" if exists
    # we want to be generous here to facilitate IO
    if split == "val" and "validation" in available:
        return "validation"

    raise ValueError(f"Split {split!r} not found in dataset: {available}")


def load_local(source: str, split: str | None = None) -> Dataset | None:
    logger.debug(f"Loading dataset from local disk: {source} (split={split})")
    try:
        d = load_from_disk(source)
    except Exception:
        return None

    if isinstance(d, Dataset):
        return d

    return d[resolve_split(split, list(map(str, d.keys())))]


def load_remote(
    source: str, split: str | None = None, args: "ProcArgs | None" = None
) -> Dataset:
    """Load a dataset from Hugging Face Hub.

    Args:
        source (str): dataset ID in the format of `org/data[@subset][:split]`.
        split (str | None, optional): dataset split name. Defaults to None.
        args (ProcArgs | None, optional): processing arguments. Defaults to None.

    Returns:
        Dataset: the loaded dataset.
    """
    logger.debug(f"Loading dataset from Hugging Face Hub: {source} (split={split})")
    source, subset, override_split = resolve_remote(source)

    d = load_dataset(
        source,
        subset,
        split=resolve_split(
            override_split or split, get_dataset_split_names(source, subset)
        ),
        num_proc=None if args is None else args.num_proc,
    )
    return d


def load_file(path: Path) -> Dataset:
    logger.debug(f"Loading dataset from file: {path}")
    if path.suffix not in _FORMATS:
        raise ValueError(f"Unsupported file format: {path.suffix!r}")

    return load_dataset(_FORMATS[path.suffix], data_files=[str(path)], split="train")


def adaptive_load_dataset(
    source: str,
    split: str | None = None,
    args: "ProcArgs | None" = None,
) -> Dataset:
    path = Path(source)

    d = None

    # non-local -> HF remote
    if not path.exists():
        if split is None:
            raise ValueError("Split must be specified for remote datasets.")
        d = load_remote(source, split, args)

    # local -> 1. dir 2. file
    elif path.is_dir():
        d = load_local(source, split)
        # try to load from locally hf-downloaded datasets
        if d is None and split is not None:
            d = load_remote(source, split, args)

    elif path.is_file():
        d = load_file(path)

    if d is None:
        raise ValueError(f"Failed to load dataset: source={source!r}, split={split!r}")

    # not randomly shuffled
    if args is not None and args.seed is not None:
        d = d.shuffle(seed=None if args.seed < 0 else args.seed)
    if args is not None and args.max_samples is not None:
        d = d.select(range(min(args.max_samples, len(d))))
    return d
