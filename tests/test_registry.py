import uuid
from pathlib import Path

import pytest
from datasets import Dataset

from prep.args import LoadArgs
from prep.registry import FormatterArgs, register_loader, status_table


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def test_formatter_args_find_splits_uses_each_registered_split(tmp_path: Path) -> None:
    data_id = f"dataset-{uuid.uuid4().hex}"

    @register_loader(data_id, "sft", "train", "src")
    def _train_loader(path: str, split: str, loadargs: LoadArgs):
        return None

    @register_loader(data_id, "sft", "test", "src")
    def _test_loader(path: str, split: str, loadargs: LoadArgs):
        return None

    args = FormatterArgs(data_id=data_id, target_format="sft", split="train")
    args.save_path(tmp_path, parquet=True).parent.mkdir(parents=True, exist_ok=True)
    test_path = FormatterArgs(
        data_id=data_id,
        target_format="sft",
        split="test",
    ).save_path(tmp_path, parquet=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("ok", encoding="utf-8")

    assert args.find_splits(tmp_path) == {"train": False, "val": None, "test": True}


def test_formatter_pipeline_load_casts_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_id = f"pipeline-{uuid.uuid4().hex}"
    rows = [
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

    @register_loader(data_id, "sft", "train", "src")
    def _loader(path: str, split: str, loadargs: LoadArgs):
        return _dataset_from_rows(rows)

    pipeline = FormatterArgs(
        data_id=data_id,
        target_format="sft",
        split="train",
    ).pipeline
    validated: dict[str, object] = {}

    monkeypatch.setattr(
        "prep.registry.validate_openai_messages",
        lambda messages, expected_n_img, img_tag: (
            validated.update(
                messages=messages,
                expected_n_img=expected_n_img,
                img_tag=img_tag,
            )
            or True
        ),
    )

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
    assert validated["expected_n_img"] == 0


def test_status_table_reports_registered_dataset(tmp_path: Path) -> None:
    data_id = f"status-{uuid.uuid4().hex}"

    @register_loader(data_id, "verl", "train", "hf/source")
    def _loader(path: str, split: str, loadargs: LoadArgs):
        return None

    table = status_table(tmp_path)
    rendered = [column.header for column in table.columns]

    assert rendered[:4] == ["Dataset ID", "Data Format", "Default Source", "Local Path"]
    assert any(cell == data_id for column in table.columns for cell in column._cells)
