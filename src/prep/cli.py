from pathlib import Path

import typer

from .args import LoadArgs, RuntimeArgs, _DataFormat, _Split
from .registry import FormatterArgs, status_table

_SAVE = "Save Options"
_HF = "HF Upload Options"
_AUTO = "Auto Conversion Options"


def prep_dataset(
    target_format: _DataFormat = typer.Argument(..., help="Target format."),
    pipeline_id: str = typer.Argument(..., help="Formatter Pipeline ID."),
    split: _Split = typer.Argument(..., help="Dataset split to convert."),
    *,
    src: str | None = typer.Option(
        None, envvar="SRC", help="Override source to load from (HF/local)."
    ),
    show: int = typer.Option(3, envvar="SHOW", help="Preview first n samples."),
    nproc: int = typer.Option(default=16, envvar="NPROC", help="Number of workers."),
    save: bool = typer.Option(True, envvar="SAVE", rich_help_panel=_SAVE),
    save_dir: Path = typer.Option(
        Path("~/.cache/prep").expanduser(), envvar="SAVE_DIR", rich_help_panel=_SAVE
    ),
    save_parq: bool = typer.Option(
        True, envvar="SAVE_PARQ", help="Output parquets/arrow.", rich_help_panel=_SAVE
    ),
    hf: bool = typer.Option(False, envvar="HF", rich_help_panel=_HF),
    hf_repo: str | None = typer.Option(None, envvar="HF_REPO", rich_help_panel=_HF),
    hf_subset: str | None = typer.Option(None, envvar="HF_SUBSET", rich_help_panel=_HF),
    hf_private: bool = typer.Option(True, envvar="HF_PRIVATE", rich_help_panel=_HF),
    q_cols: list[str] = typer.Option(
        default=["question", "Question", "problem"],
        envvar="Q_COLS",
        rich_help_panel=_AUTO,
    ),
    q_template: str = typer.Option(
        default="{question}",
        envvar="Q_TEMP",
        rich_help_panel=_AUTO,
    ),
    a_cols: list[str] = typer.Option(
        default=["answer", "Answer", "solution", "label", "caption", "correct_answer"],
        envvar="A_COLS",
        rich_help_panel=_AUTO,
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
        d, save_path=pipeline.save_path(runtime.save_dir, runtime.save_parquet)
    )
    runtime.do_upload(d, split=pipeline.split)


def prep():
    typer.run(prep_dataset)


def status():
    def data_status(
        save_dir: Path = typer.Option(
            default=Path("~/.cache/prep").expanduser(), envvar="SAVE_DIR"
        ),
    ):
        from rich.console import Console

        console = Console()
        console.print(status_table(save_dir))

    typer.run(data_status)
