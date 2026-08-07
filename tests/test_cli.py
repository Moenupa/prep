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
        def __init__(self, data_id: str, target_format: str, split: str) -> None:
            calls.append(("formatter_args", data_id, target_format, split))
            self.pipeline = FakePipeline()

    monkeypatch.setattr(cli, "FormatterArgs", FakeFormatterArgs)
    monkeypatch.setattr(
        cli.RuntimeArgs,
        "peek",
        lambda self, dataset: calls.append(("peek", dataset)),
    )
    monkeypatch.setattr(
        cli.RuntimeArgs,
        "save",
        lambda self, dataset, save_path: calls.append(("save", dataset, save_path)),
    )
    monkeypatch.setattr(
        cli.RuntimeArgs,
        "upload",
        lambda self, dataset, split: calls.append(("upload", dataset, split)),
    )

    cli.prep_dataset(
        target_format="sft",
        data_id="demo",
        split="train",
        src="local",
        show=2,
        save_dir=tmp_path,
        save_parq=True,
        hf=True,
        hf_repo="repo",
        hf_subset="subset",
        hf_private=False,
        dry=False,
        nproc=3,
        auto_q_cols=["question"],
        auto_q_template="{question}",
        auto_a_cols=["answer"],
    )

    assert calls == [
        ("formatter_args", "demo", "sft", "train"),
        ("load", "local", 3),
        ("peek", "dataset"),
        ("upload", "dataset", "train"),
        ("save_path", tmp_path, True),
        ("save", "dataset", tmp_path / "train.parquet"),
    ]
