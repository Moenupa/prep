from pathlib import Path

import typer

from .api import FormatterPipeline, OutputActions, PathIO, ProcArgs
from .api.types import _DataFormat, _Split
from .constants import (
    DEFAULT_ACOLS,
    DEFAULT_ATEMP,
    DEFAULT_OPCOLS,
    DEFAULT_QCOLS,
    DEFAULT_QTEMP,
)

_SAVE = "Save Options"
_HF = "HF Upload Options"
_AUTO = "Auto Conversion Options"
app = typer.Typer()


@app.command()
def prep(
    target_format: _DataFormat = typer.Argument(..., help="Target format."),
    pipeline_id: str = typer.Argument(..., help="Formatter Pipeline ID."),
    split: _Split = typer.Argument("train", help="Split to convert."),
    *,
    src: str | None = typer.Option(None, envvar="SRC", help="Source, HF/local path."),
    show: int = typer.Option(3, envvar="SHOW", help="Preview first n samples."),
    nproc: int = typer.Option(16, envvar="NPROC", help="Workers for processing."),
    save: bool | None = typer.Option(None, envvar="SAVE", rich_help_panel=_SAVE),
    save_root: Path = typer.Option(
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
    max_samples: int | None = typer.Option(
        None, envvar="MAX_SAMPLES", rich_help_panel=_AUTO
    ),
    q_cols: list[str] = typer.Option(
        default=DEFAULT_QCOLS, envvar="Q_COLS", rich_help_panel=_AUTO
    ),
    q_template: str = typer.Option(
        default=DEFAULT_QTEMP, envvar="Q_TEMP", rich_help_panel=_AUTO
    ),
    op_cols: list[str] = typer.Option(
        default=DEFAULT_OPCOLS, envvar="OP_COLS", rich_help_panel=_AUTO
    ),
    a_cols: list[str] = typer.Option(
        default=DEFAULT_ACOLS, envvar="A_COLS", rich_help_panel=_AUTO
    ),
    a_template: str = typer.Option(
        default=DEFAULT_ATEMP, envvar="A_TEMP", rich_help_panel=_AUTO
    ),
    verl_ability: str = typer.Option(
        default="math", envvar="VERL_ABILITY", rich_help_panel=_AUTO
    ),
    verl_style: str = typer.Option(
        default="rule", envvar="VERL_STYLE", rich_help_panel=_AUTO
    ),
):
    pipeline = FormatterPipeline.get(
        id_=pipeline_id,
        target_format=target_format,
        split=split,
    )
    d = pipeline.load(
        src,
        ProcArgs(
            num_proc=nproc,
            question_cols=q_cols,
            question_template=q_template,
            option_cols=op_cols,
            answer_cols=a_cols,
            answer_template=a_template,
            verl_ability=verl_ability,
            verl_style=verl_style,
            show_first_n=show,
            max_samples=max_samples,
        ),
    )
    action = OutputActions(
        save=save,
        save_root=save_root.expanduser(),
        save_parquet=save_parq,
        hf=hf,
        hf_repo=hf_repo or pipeline_id,
        hf_subset=hf_subset,
        hf_private=hf_private,
    )
    action.do_save(d, pipeline=pipeline, nproc=save_nproc, id_override=pipeline_id)
    action.do_upload(d, split=pipeline.split, nproc=hf_nproc)


@app.command()
def ppls(
    save_root: Path = typer.Argument(default=Path("out"), envvar="SAVE_DIR"),
    as_json: bool = False,
):
    from rich.console import Console

    console = Console()
    console.print(f"Showing results under {save_root.as_posix()!r}")
    if as_json:
        console.print(*PathIO(save_root).status_dicts())
    else:
        console.print(*PathIO(save_root).status_tables())


def ppls_cli():
    typer.run(ppls)


def prep_cli():
    typer.run(prep)
