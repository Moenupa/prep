from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .formatter import FormatterPipeline, get_registered_pipelines
from .types import DataFormat, Split, get_valid_formats, get_valid_splits

if TYPE_CHECKING:
    from collections.abc import Generator

    from rich.table import Table


@dataclass(frozen=True)
class PathIO:
    save_root: Path

    def per_fmt_status(
        self,
        target_fmt: DataFormat,
        registered_ids: dict[tuple[DataFormat, str], str | None],
    ) -> dict[str, list]:
        out: dict[str, list] = defaultdict(list)
        for (fmt, id_), default_src in sorted(registered_ids.items()):
            if fmt != target_fmt:
                continue

            local_dir = PathIO.dataset_dir(self.save_root, fmt, id_)
            # show unregistered as None (N.A.) instead of False (not processed)
            local_splits = self.scan_splits(
                local_dir
            ) | FormatterPipeline.unregistered_splits(id_=id_, target_format=fmt)

            if not local_dir.exists() or not any(local_dir.glob("*")):
                local_dir = None

            out["ID"].append(id_)
            out["Default Source"].append(default_src or "")
            out["Local Path"].append(str(local_dir or ""))
            out["Size"].append(self.size(local_dir))
            for split in get_valid_splits():
                out[split].append(local_splits.get(split))
        return out

    def status_dicts(
        self, filter_format: str = "*"
    ) -> "Generator[tuple[str, dict[str, list]], None, None]":
        from fnmatch import fnmatch

        registered_ids = get_registered_pipelines()
        for cur_fmt in get_valid_formats():
            if fnmatch(cur_fmt, filter_format):
                yield cur_fmt, self.per_fmt_status(cur_fmt, registered_ids)

        unregistered_items = {
            k: v
            for k, v in self.scan_datasets(self.save_root).items()
            if k not in registered_ids
        }
        if not unregistered_items or fnmatch("others", filter_format) is False:
            return

        out: dict[str, list] = defaultdict(list)
        for (fmt, id_), local_dir in sorted(unregistered_items.items()):
            local_splits = self.scan_splits(local_dir)
            out["ID"].append(id_)
            out["Format"].append(fmt)
            out["Local Path"].append(str(local_dir))
            out["Size"].append(self.size(local_dir))
            for split in get_valid_splits():
                out[split].append(local_splits.get(split))
        yield "others", out

    def status_tables(self, filter_format: str = "*") -> "Generator[Table, None, None]":
        from rich.table import Table

        for title, out in self.status_dicts(filter_format=filter_format):
            table = Table(show_header=True, title=title, header_style="bold magenta")
            for colname, style in zip(
                out.keys(),
                ["cyan", "green", "blue", "yellow"]
                + ["yellow"] * len(get_valid_splits()),
            ):
                table.add_column(
                    colname, style=style, no_wrap="path" not in colname.lower()
                )

            for i in range(len(out["ID"])):
                table.add_row(
                    *(
                        {True: "✔", False: "✖", None: ""}.get(v[i], str(v[i]))
                        for v in out.values()
                    ),
                    style=None if out["Local Path"][i] else "dim",
                )
            yield table

    def default_save_path(
        self,
        pipeline: FormatterPipeline,
        as_parquet: bool,
        id_override: str | None = None,
    ) -> Path:
        return PathIO.dataset_path(
            dataset_dir=self.save_root
            / pipeline.target_format
            / (id_override or pipeline.id_),
            split=pipeline.split,
            as_parquet=as_parquet,
        )

    @staticmethod
    def is_overwrite(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file():
            return True
        if any(p for p in path.iterdir() if p.is_file()):
            return True
        return False

    @staticmethod
    def dataset_path(dataset_dir: Path, split: Split, as_parquet: bool) -> Path:
        save_to = dataset_dir / split
        if as_parquet:
            return save_to.with_suffix(".parquet")

        return save_to

    @staticmethod
    def dataset_dir(save_root: Path, fmt: DataFormat, dataset_id: str) -> Path:
        return save_root / fmt / dataset_id

    @staticmethod
    def scan_datasets(save_root: Path) -> dict[tuple[DataFormat, str], Path]:
        return {
            # out/verl/xxx -> (verl, xxx): out/verl/xxx
            (each_format, dataset_dir.stem): dataset_dir
            for each_format in get_valid_formats()
            if (save_root / each_format).exists()
            for dataset_dir in (save_root / each_format).iterdir()
            if dataset_dir.is_dir()
        }

    @staticmethod
    def scan_splits(dataset_dir: Path) -> dict[Split, bool | None]:
        return {
            split: (
                PathIO.dataset_path(dataset_dir, split, as_parquet=True).exists()
                or any(
                    PathIO.dataset_path(dataset_dir, split, as_parquet=False).glob("*")
                )
            )
            for split in get_valid_splits()
        }

    @staticmethod
    def size(dataset_dir: Path | None) -> str:
        if dataset_dir is None:
            return ""
        from rich.filesize import decimal

        return decimal(
            sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()),
            precision=0,
        )
