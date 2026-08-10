from pathlib import Path

from prep import cli
from prep.args import LoadArgs


def test_cli_prep_dataset_orchestrates_pipeline(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple] = []

    class FakePipeline:
        split = "train"

        def load(self, override_src: str | None, loadargs: LoadArgs):
            calls.append(("load", override_src, loadargs.num_proc))
            return "dataset"

        def save_path(self, save_dir: Path, save_parquet: bool) -> Path:
            calls.append(("save_path", save_dir, save_parquet))
            return save_dir / "train.parquet"

    class FakeFormatterArgs:
        def __init__(self, id_: str, target_format: str, split: str) -> None:
            calls.append(("formatter_args", id_, target_format, split))
            self.pipeline = FakePipeline()

    monkeypatch.setattr(cli, "FormatterArgs", FakeFormatterArgs)
    monkeypatch.setattr(
        cli.RuntimeArgs,
        "do_save",
        lambda self, dataset, save_path: calls.append(("do_save", dataset, save_path)),
    )
    monkeypatch.setattr(
        cli.RuntimeArgs,
        "do_upload",
        lambda self, dataset, split: calls.append(("do_upload", dataset, split)),
    )

    cli.prep_dataset(
        target_format="sft",
        pipeline_id="demo",
        split="train",
        src="local",
        show=2,
        save_dir=tmp_path,
        save_parq=True,
        hf=True,
        hf_repo="repo",
        hf_subset="subset",
        hf_private=False,
        save=False,
        nproc=3,
        q_cols=["question"],
        q_template="{question}",
        a_cols=["answer"],
    )

    assert calls == [
        ("formatter_args", "demo", "sft", "train"),
        ("load", "local", 3),
        ("save_path", tmp_path, True),
        ("do_save", "dataset", tmp_path / "train.parquet"),
        ("do_upload", "dataset", "train"),
    ]
