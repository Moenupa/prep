from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from datasets import get_dataset_config_names, get_dataset_split_names
from rich import print as rprint

from ..constants import HF_PREFIX, SAVE_PREFIX, SKIP
from .formatutils import iter_images, write_to_file
from .log import get_logger
from .pathio import PathIO

if TYPE_CHECKING:
    from datasets import Dataset

    from .formatter import FormatterPipeline
    from .types import Split


logger = get_logger(__name__)


def _hf_target_exists(
    repo_id: str,
    config_name: str,
    split: str,
) -> bool:
    try:
        from huggingface_hub import HfApi

        if not HfApi().repo_exists(repo_id, repo_type="dataset"):
            return False

        configs = get_dataset_config_names(repo_id)
        return config_name in configs and split in get_dataset_split_names(
            repo_id, config_name
        )
    except Exception as exc:
        logger.warning(exc)
        logger.warning(f"Unable to inspect HF repo {repo_id}. Treating as existing.")
        return True


def _resolve_hf_target(
    repo_id: str,
    config_name: str,
    split: str,
) -> tuple[str, str, str] | None:
    def hf_target(inp: str) -> tuple[str, str, str] | None:
        if inp.strip() == SKIP:
            return None

        try:
            _1, _2, _3 = inp.split(" ")
            return _1, _2, _3
        except ValueError:
            raise typer.BadParameter("Input must be in the format 'REPO SUBSET SPLIT'")

    while True:
        target = f"{repo_id} {config_name} {split}"

        act = (
            "Overwrite" if _hf_target_exists(repo_id, config_name, split) else "Upload"
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


@dataclass(frozen=True)
class DataIO(PathIO):
    save: bool | None
    save_parquet: bool

    hf: bool | None
    hf_repo: str
    hf_subset: str | None
    hf_private: bool

    preview_first_n: int | None = None
    preview_last_n: int | None = None

    def do_preview(
        self,
        d: "Dataset",
        pipeline: "FormatterPipeline",
        id_override: str | None = None,
    ) -> None:
        rprint(d)

        save_root = self.default_save_path(
            pipeline, as_parquet=False, id_override=id_override
        )
        save_root.mkdir(parents=True, exist_ok=True)

        for i in chain(
            range(min(self.preview_first_n or 0, len(d))),
            range(max(len(d) - (self.preview_last_n or 0), 0), len(d)),
        ):
            try:
                row = d[i]
                image_paths = [
                    str(write_to_file(img, save_root / f"{i:06d}_{j:02d}"))
                    for j, img in enumerate(iter_images(row))
                ]

                rprint(f"[bold]Sample {i}[/bold]:" + (
                    f" (images {image_paths})" if image_paths else ""
                ), row)
            except Exception as exc:
                logger.warning(f"Failed to preview {i}th sample: {exc}")
                continue

    def do_save(
        self,
        d: "Dataset",
        pipeline: "FormatterPipeline",
        nproc: int | None = None,
        id_override: str | None = None,
        dry_run: bool = False,
    ):
        save_path = self.default_save_path(
            pipeline, self.save_parquet, id_override=id_override
        )
        typer.echo(f"{SAVE_PREFIX}About to save to disk -> {save_path!r}...")
        match self.save:
            # if --save, force saving without prompt
            case True:
                pass
            # if --no-save is passed, skip saving as a whole.
            case False:
                typer.echo(
                    f"{SAVE_PREFIX}Not saved (pass `--save` to override or `export UI=1` to enable interactive prompts)"
                )
                return
            case None:
                resolved_path = _resolve_save_path(save_path)
                if resolved_path is None:
                    typer.echo(f"{SAVE_PREFIX}Not saved")
                    return
                save_path = resolved_path

        typer.echo(f"{SAVE_PREFIX}Saving to disk -> {save_path!r}...")
        if dry_run:
            return
        if self.save_parquet:
            logger.warning(
                "Saving parquet may not preserve media features as binary objects."
            )
            d.to_parquet(save_path)
        else:
            d.save_to_disk(save_path, num_proc=nproc)

    def do_upload(
        self,
        d: "Dataset",
        split: "Split | str",
        nproc: int | None = None,
        dry_run: bool = False,
    ):
        repo_id = self.hf_repo
        subset = self.hf_subset or "default"
        split = {"val": "validation"}.get(split, split)

        typer.echo(
            f"{HF_PREFIX}About to upload to HF -> {repo_id!r}"
            f" (subset={subset!r}, split={split!r}, private={self.hf_private})..."
        )
        match self.hf:
            # if --hf, force upload without prompt
            case True:
                pass
            # if --no-hf is passed, skip uploading as a whole.
            case False:
                typer.echo(
                    f"{HF_PREFIX}Not uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"
                )
                return
            case None:
                target = _resolve_hf_target(
                    repo_id=repo_id, config_name=subset, split=split
                )
                if target is None:
                    typer.echo(f"{HF_PREFIX}Not uploaded")
                    return
                repo_id, subset, split = target

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
