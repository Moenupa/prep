import shutil
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

import filetype
import typer
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
)
from huggingface_hub import HfApi
from PIL import Image as PILImage
from rich.filesize import decimal
from rich.table import Table

from ..constants import HF_PREFIX, PREVIEW_PREFIX, SAVE_PREFIX, SKIP, WARN_PREFIX
from .formatter import FormatterPipeline, get_registered_pipelines
from .formatutils import iter_images
from .log import get_logger
from .types import DataFormat, Split, get_valid_formats, get_valid_splits

if TYPE_CHECKING:
    from collections.abc import Generator


logger = get_logger(__name__)


def guess_ext(raw: bytes, src_path: str | None) -> str:
    ext = filetype.guess_extension(raw)
    if ext is None and src_path is not None:
        ext = Path(src_path).suffix.lower().lstrip(".")
    if ext is None:
        raise ValueError("Cannot guess file extension from raw bytes or source path")

    return ext


def _write_preview_image(entry: Any, dest: Path) -> Path | None:
    try:
        if isinstance(entry, PILImage.Image):
            ext = (entry.format or "PNG").lower()
            if ext == "jpeg":
                ext = "jpg"
            img = entry
            if ext == "jpg" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            out = dest.with_suffix(f".{ext}")
            img.save(out)
            return out

        if isinstance(entry, dict):
            raw, src_path = entry.get("bytes"), entry.get("path")
            if raw:
                out = dest.with_suffix(f".{guess_ext(raw, src_path)}")
                out.write_bytes(raw)
                return out
            if src_path and Path(src_path).exists():
                src = Path(src_path)
                out = dest.with_suffix(src.suffix.lower())
                shutil.copyfile(src, out)
                return out

        if isinstance(entry, (str, Path)) and Path(entry).exists():
            src = Path(entry)
            out = dest.with_suffix(src.suffix.lower())
            shutil.copyfile(src, out)
            return out
    except Exception as exc:
        logger.warning(f"{WARN_PREFIX}Failed to dump preview image: {exc}")
    return None


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
        return decimal(
            sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()),
            precision=0,
        )


@dataclass(frozen=True)
class OutputActions(PathIO):
    save: bool | None
    save_parquet: bool
    save_preview: int | None

    hf: bool | None
    hf_repo: str
    hf_subset: str | None
    hf_private: bool

    interactive: bool = False

    def do_dump(
        self,
        d: "Dataset",
        pipeline: FormatterPipeline,
        id_override: str | None = None,
    ) -> None:
        """Dump images from the first ``self.preview`` rows for visual inspection.

        Since image columns are stored as opaque binary blobs in parquet/arrow
        outputs, this writes a few representative image files to disk under the
        default save path so an agent or user can directly inspect them.

        Args:
            d: The formatted dataset to preview.
            pipeline: The pipeline used to produce ``d`` (used for the output path).
            id_override: Overrides the pipeline ID in the output path, mirroring
                the behavior of ``OutputActions.do_save``.
        """
        if self.save_preview is None or self.save_preview <= 0:
            return
        dump_dir = self.default_save_path(
            pipeline, as_parquet=False, id_override=id_override
        )
        dump_dir.mkdir(parents=True, exist_ok=True)

        for i in range(min(self.save_preview, len(d))):
            try:
                row = d[i]
            except Exception as exc:
                logger.warning(
                    f"{WARN_PREFIX}Failed to load sample {i} for preview: {exc}"
                )
                continue
            for j, img in enumerate(iter_images(row)):
                dest = dump_dir / f"{i:06d}_{j:02d}"
                path = _write_preview_image(img, dest)
                if path is not None:
                    typer.echo(f"{PREVIEW_PREFIX}Preview image saved: {path}")

    @staticmethod
    def _resolve_save_path(save_path: Path) -> Path | None:
        def save_target(e: str) -> Path | None:
            if e.strip() == SKIP:
                return None

            try:
                p = Path(e).expanduser()
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception as exc:
                raise typer.BadParameter(f"Invalid path: {e!r} {exc}")

        while True:
            act = "Overwrite" if PathIO.is_overwrite(save_path) else "Save"
            new_path = typer.prompt(
                f"{SAVE_PREFIX}{act}? ({SKIP!r} to skip"
                ", <enter> to confirm, or enter another path to change)",
                default=str(save_path),
                value_proc=save_target,
                err=True,
            )
            if new_path is None:
                return None

            if new_path == save_path:
                return save_path
            save_path = new_path

    def do_save(
        self,
        d: "Dataset",
        pipeline: FormatterPipeline,
        nproc: int | None = None,
        id_override: str | None = None,
        dry_run: bool = False,
    ):
        save_path = self.default_save_path(
            pipeline, self.save_parquet, id_override=id_override
        )
        typer.echo(f"{SAVE_PREFIX}About to save to disk -> {save_path!r}")
        match self.save:
            # if --save, force saving without prompt
            case True:
                pass
            # if --no-save is passed, skip saving as a whole.
            case False:
                typer.echo(f"{SAVE_PREFIX}Not saving (--no-save passed)")
                return
            # otherwise, go into interactive mode and figure out save_path
            case None if self.interactive:
                resolved_path = self._resolve_save_path(save_path)
                if resolved_path is None:
                    typer.echo(f"{SAVE_PREFIX}Not saving")
                    return
                save_path = resolved_path
            case None:
                # non-interactive mode: skip saving
                typer.echo(f"{SAVE_PREFIX}Skipping (non-interactive mode)")
                return

        typer.echo(f"{SAVE_PREFIX}Saving to disk -> {save_path!r}")
        if dry_run:
            return
        if self.save_parquet:
            logger.warning(
                "Saving parquet may not preserve media features as binary objects."
            )
            d.to_parquet(save_path)
        else:
            d.save_to_disk(save_path, num_proc=nproc)

    @staticmethod
    def _hf_target_exists(
        repo_id: str,
        config_name: str,
        split: str,
    ) -> bool:
        try:
            if not HfApi().repo_exists(repo_id, repo_type="dataset"):
                return False

            configs = get_dataset_config_names(repo_id)
            return config_name in configs and split in get_dataset_split_names(
                repo_id, config_name
            )
        except Exception as exc:
            logger.warning(exc)
            logger.warning(
                f"Unable to inspect HF repo {repo_id}. Treating as existing."
            )
            return True

    @staticmethod
    def _resolve_hf_target(
        repo_id: str,
        config_name: str,
        split: str,
    ) -> tuple[str, str, str] | None:
        import typer

        def hf_target(inp: str) -> tuple[str, str, str] | None:
            if inp.strip() == SKIP:
                return None

            try:
                _1, _2, _3 = inp.split(" ")
                return _1, _2, _3
            except ValueError:
                raise typer.BadParameter(
                    "Input must be in the format 'REPO SUBSET SPLIT'"
                )

        while True:
            target = f"{repo_id} {config_name} {split}"

            act = (
                "Overwrite"
                if OutputActions._hf_target_exists(repo_id, config_name, split)
                else "Upload"
            )
            new_target = typer.prompt(
                f"{HF_PREFIX}HF {act} ({SKIP!r} to skip"
                ", <enter> to confirm, or modify in format 'REPO SUBSET SPLIT')",
                default=target,
                value_proc=hf_target,
                err=True,
            )
            if new_target is None:
                return None

            repo_id, config_name, split = new_target
            if target == f"{repo_id} {config_name} {split}":
                return repo_id, config_name, split

    def do_upload(
        self,
        d: "Dataset",
        split: Split | str,
        nproc: int | None = None,
        dry_run: bool = False,
    ):
        import typer

        repo_id = self.hf_repo
        subset = self.hf_subset or "default"
        split = {"val": "validation"}.get(split, split)

        typer.echo(
            f"{HF_PREFIX}About to upload to HF -> {repo_id!r}"
            f" (subset={subset!r}, split={split!r}, private={self.hf_private})"
        )
        match self.hf:
            # if --hf, force upload without prompt
            case True:
                pass
            # if --no-hf is passed, skip uploading as a whole.
            case False:
                typer.echo(f"{HF_PREFIX}Not uploading (--no-hf passed)")
                return
            case None if self.interactive:
                target = self._resolve_hf_target(
                    repo_id=repo_id, config_name=subset, split=split
                )
                if target is None:
                    typer.echo(f"{HF_PREFIX}Not uploading")
                    return
                repo_id, subset, split = target
            case None:
                # non-interactive mode: skip uploading
                typer.echo(f"{HF_PREFIX}Skipping (non-interactive mode)")
                return

        typer.echo(
            f"{HF_PREFIX}Uploading to HF -> {repo_id!r}"
            f" (subset={subset!r}, split={split!r}, private={self.hf_private})"
        )
        if dry_run:
            return
        d.push_to_hub(
            repo_id,
            config_name=subset,
            split=split,
            private=self.hf_private,
            num_proc=nproc,
        )
