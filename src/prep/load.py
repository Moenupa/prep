from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

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


def adaptive_load_dataset(source: str, split: str) -> Dataset:
    path = Path(source)

    # non-local -> HF remote
    if not path.exists():
        return load_dataset(source, split=split)

    # local -> 1. dir 2. file
    elif path.is_dir():
        try:
            d = load_from_disk(source)
            if isinstance(d, DatasetDict):
                return d[split]
            if isinstance(d, Dataset):
                return d
            raise RuntimeError(f"Loaded not Dataset nor DatasetDict: {type(d)} {d}")
        except Exception as e:
            raise ValueError(f"Failed to load from local folder {source!r}") from e
    elif path.is_file():
        if path.suffix not in _FORMATS:
            raise ValueError(f"Unsupported file format: {path.suffix!r}")

        d = load_dataset(_FORMATS[path.suffix], data_files=[source], split="train")
        return d

    raise ValueError(f"Invalid source path: {source!r}")
