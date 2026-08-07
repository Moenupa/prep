from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

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


def _dataset(path: str) -> Dataset | None:
    try:
        return Dataset.load_from_disk(path)
    except Exception:
        return None


def _dataset_dict(path: str, split: str) -> Dataset | None:
    try:
        return DatasetDict.load_from_disk(path)[split]
    except Exception:
        return None


def adaptive_load_dataset(source: str, split: str) -> Dataset:
    path = Path(source)
    if not path.exists():
        return load_dataset(source, split=split)
    elif path.is_dir():
        d = _dataset(source) or _dataset_dict(source, split)
        if d is None:
            raise ValueError(f"Failed to load from local folder {source!r}")
        return d
    elif path.is_file():
        if path.suffix not in _FORMATS:
            raise ValueError(f"Unsupported file format: {path.suffix!r}")

        d = load_dataset(_FORMATS[path.suffix], data_files=[source], split="train")
        return d

    raise ValueError(f"Invalid source path: {source!r}")
