from pathlib import Path

import typer

from .args import LoadArgs, RuntimeArgs, _DataFormat, _Split
from .registry import FormatterArgs, status_table


def prep_dataset(
    target_format: _DataFormat = typer.Argument(
        ..., help="Target format to convert to."
    ),
    data_id: str = typer.Argument(..., help="Dataset ID."),
    split: _Split = typer.Argument(..., help="Dataset split to convert."),
    src: str | None = typer.Option(
        None, envvar="SRC", help="Override source to load from (HF/local)."
    ),
    show: int = typer.Option(
        3, envvar="SHOW", help="Preview first n samples after formatting."
    ),
    save_dir: Path = typer.Option(
        Path("~/.cache/prep").expanduser(), envvar="SAVE_DIR"
    ),
    save_parq: bool = typer.Option(
        True, envvar="SAVE_PARQ", help="Save as parquets/arrow."
    ),
    hf: bool = typer.Option(
        False, envvar="HF", help="Upload HF. If False, save locally."
    ),
    hf_repo: str | None = typer.Option(
        None, envvar="HF_REPO", help="HF Repo. If None, use data_id."
    ),
    hf_subset: str | None = typer.Option(
        None, envvar="HF_SUBSET", help="HF Subset. If None, 'default'."
    ),
    hf_private: bool = typer.Option(
        True, envvar="HF_PRIVATE", help="HF repo as private."
    ),
    dry: bool = typer.Option(False, envvar="DRY", help="If True, no save/upload."),
    nproc: int = typer.Option(default=16, envvar="NPROC", help="Number of workers."),
    auto_q_cols: list[str] = typer.Option(
        default=["question", "Question", "problem"], envvar="Q_COLS"
    ),
    auto_q_template: str = typer.Option(default="{question}", envvar="Q_TEMPLATE"),
    auto_a_cols: list[str] = typer.Option(
        default=["answer", "Answer", "solution", "label"], envvar="A_COLS"
    ),
):
    pipeline = FormatterArgs(
        data_id=data_id,
        target_format=target_format,
        split=split,
    ).pipeline
    runtime = RuntimeArgs(
        override_src=src,
        show_first_n=show,
        save_dir=save_dir,
        save_parquet=save_parq,
        hf=hf,
        hf_repo=hf_repo or data_id,
        hf_subset=hf_subset,
        hf_private=hf_private,
        dry_run=dry,
    )
    loadargs = LoadArgs(
        num_proc=nproc,
        question_cols=auto_q_cols,
        question_template=auto_q_template,
        answer_cols=auto_a_cols,
    )
    d = pipeline.load(runtime.override_src, loadargs)
    runtime.peek(d)

    # upload to hf
    if runtime.hf:
        runtime.upload(d, split=pipeline.split)

    # save to disk
    runtime.save(d, pipeline.save_path(runtime.save_dir, runtime.save_parquet))


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
