from pathlib import Path

import typer

from . import formatter  # noqa: F401
from .constants import get_logger
from .registry import DataFormat, Split, load, peek, save_to, status_table

logger = get_logger(__name__)


def prep_dataset(
    fmt: DataFormat,
    data_id: str,
    split: Split,
    from_local: Path | None = typer.Option(
        None, help="Source has a local path or can not be directly loaded from HF"
    ),
    show: int = typer.Option(default=3, help="Showcase first n samples after loading."),
    save_dir: Path = typer.Option(
        default=Path("~/.cache/prep").expanduser(), envvar="PREP_DIR"
    ),
    save_parq: bool = typer.Option(
        True, help="Whether to save as parquet. If False, save arrow."
    ),
    hf_subset: str | None = typer.Option(
        None, help="HF Subset (e.g., 'default'). If None, do not upload."
    ),
    hf_private: bool = typer.Option(
        True,
        help="Whether to upload as private. Works only if hf_subset is not None.",
    ),
    dry: bool = typer.Option(
        False, envvar="DRY", help="If True, do not save or upload, just show the info."
    ),
):
    d = load(data_id, fmt, split, from_local.as_posix() if from_local else None)
    peek(d, show)

    # upload to hf
    if hf_subset is not None:
        logger.info(
            f"☁️\tUploading to {data_id!r} (subset={hf_subset!r}, split={split!r}, private={hf_private})"
        )
        if dry:
            return
        d.push_to_hub(data_id, hf_subset, split=split, private=hf_private)
        return

    # save to disk
    save_to(d, data_id, fmt, split, dry_run=dry, save_dir=save_dir, parquet=save_parq)


def prep():
    typer.run(prep_dataset)


def status():
    def data_status(
        save_dir: Path = typer.Option(
            default=Path("~/.cache/prep").expanduser(), envvar="PREP_DIR"
        ),
    ):
        from rich.console import Console

        console = Console()
        console.print(status_table(save_dir))

    typer.run(data_status)
