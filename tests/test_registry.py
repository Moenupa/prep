import uuid
from pathlib import Path

from datasets import Dataset

from prep.args import LoadArgs
from prep.registry import FormatterArgs, formatter


def test_formatter_args_find_splits_uses_each_registered_split(tmp_path: Path) -> None:
    id_ = f"dataset-{uuid.uuid4().hex}"

    @formatter(id_, "sft", "train", "src")
    def _train_loader(path: str, split: str, loadargs: LoadArgs):
        return None

    @formatter(id_, "sft", "test", "src")
    def _test_loader(path: str, split: str, loadargs: LoadArgs):
        return None

    args = FormatterArgs(id_=id_, target_format="sft", split="train")
    args.save_path(tmp_path, parquet=True).parent.mkdir(parents=True, exist_ok=True)
    test_path = FormatterArgs(
        id_=id_,
        target_format="sft",
        split="test",
    ).save_path(tmp_path, parquet=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("ok", encoding="utf-8")

    assert args.find_splits(tmp_path) == {"train": False, "val": None, "test": True}


def test_formatter_pipeline_load_casts_and_validates() -> None:
    id_ = f"pipeline-{uuid.uuid4().hex}"

    @formatter(id_, "sft", "train", "src")
    def _loader(path: str, split: str, loadargs: LoadArgs):
        return Dataset.from_list(
            [
                {
                    "images": [],
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "id": "x",
                    "extra_info": "",
                }
            ]
        )

    pipeline = FormatterArgs(
        id_=id_,
        target_format="sft",
        split="train",
    ).pipeline

    dataset = pipeline.load(
        None,
        LoadArgs(
            num_proc=1,
            question_cols=["q"],
            question_template="{question}",
            answer_cols=["a"],
        ),
    )

    assert dataset[0]["id"] == "x"
