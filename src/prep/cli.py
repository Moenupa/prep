from pathlib import Path

import typer

from .args import LoadArgs, RuntimeArgs, _DataFormat, _Split
from .constants import DEFAULT_ACOLS, DEFAULT_QCOLS
from .registry import FormatterArgs, status_table

_SAVE = "Save Options"
_HF = "HF Upload Options"
_AUTO = "Auto Conversion Options"
app = typer.Typer()


@app.command()
def prep(
    target_format: _DataFormat = typer.Argument(..., help="Target format."),
    pipeline_id: str = typer.Argument(..., help="Formatter Pipeline ID."),
    split: _Split = typer.Argument(..., help="Dataset split to convert."),
    *,
    src: str | None = typer.Option(None, envvar="SRC", help="Source, HF/local path."),
    show: int = typer.Option(3, envvar="SHOW", help="Preview first n samples."),
    nproc: int = typer.Option(16, envvar="NPROC", help="Workers for processing."),
    save: bool | None = typer.Option(None, envvar="SAVE", rich_help_panel=_SAVE),
    save_dir: Path = typer.Option(
        Path("out"), envvar="SAVE_DIR", rich_help_panel=_SAVE
    ),
    save_parq: bool = typer.Option(False, envvar="SAVE_PARQ", rich_help_panel=_SAVE),
    save_nproc: int | None = typer.Option(
        None, envvar="SAVE_NPROC", rich_help_panel=_SAVE
    ),
    hf: bool | None = typer.Option(None, envvar="HF", rich_help_panel=_HF),
    hf_repo: str | None = typer.Option(None, envvar="HF_REPO", rich_help_panel=_HF),
    hf_subset: str | None = typer.Option(None, envvar="HF_SUBSET", rich_help_panel=_HF),
    hf_private: bool = typer.Option(True, envvar="HF_PRIVATE", rich_help_panel=_HF),
    hf_nproc: int | None = typer.Option(None, envvar="HF_NPROC", rich_help_panel=_HF),
    q_cols: list[str] = typer.Option(
        default=DEFAULT_QCOLS, envvar="Q_COLS", rich_help_panel=_AUTO
    ),
    q_template: str = typer.Option(
        default="{question}", envvar="Q_TEMP", rich_help_panel=_AUTO
    ),
    a_cols: list[str] = typer.Option(
        default=DEFAULT_ACOLS, envvar="A_COLS", rich_help_panel=_AUTO
    ),
):
    pipeline = FormatterArgs(
        id_=pipeline_id,
        target_format=target_format,
        split=split,
    ).pipeline
    runtime = RuntimeArgs(
        override_src=src,
        save=save,
        save_dir=save_dir,
        save_parquet=save_parq,
        hf=hf,
        hf_repo=hf_repo or pipeline_id,
        hf_subset=hf_subset,
        hf_private=hf_private,
    )
    d = pipeline.load(
        runtime.override_src,
        LoadArgs(
            num_proc=nproc,
            question_cols=q_cols,
            question_template=q_template,
            answer_cols=a_cols,
            show_first_n=show,
        ),
    )
    runtime.do_save(
        d,
        save_path=pipeline.save_path(runtime.save_dir, runtime.save_parquet),
        nproc=save_nproc,
    )
    runtime.do_upload(d, split=pipeline.split, nproc=hf_nproc)


@app.command()
def ppls(
    save_dir: Path = typer.Argument(default=Path("out"), envvar="SAVE_DIR"),
):
    from rich.console import Console

    console = Console()
    console.print(f"Showing pipelines under {save_dir.as_posix()}")
    console.print(*status_table(save_dir))


def ppls_cli():
    typer.run(ppls)


def prep_cli():
    typer.run(prep)
