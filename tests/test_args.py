from pathlib import Path

import pytest
from datasets import Dataset

from prep.args import LoadArgs, RuntimeArgs


def test_load_args_validate_required_fields() -> None:
    with pytest.raises(ValueError, match="num_proc"):
        LoadArgs(
            num_proc=0,
            question_cols=["q"],
            question_template="{question}",
            answer_cols=["a"],
        )

    with pytest.raises(ValueError, match="question_cols"):
        LoadArgs(
            num_proc=1,
            question_cols=[],
            question_template="{question}",
            answer_cols=["a"],
        )

    with pytest.raises(ValueError, match="answer_cols"):
        LoadArgs(
            num_proc=1,
            question_cols=["q"],
            question_template="{question}",
            answer_cols=[],
        )

    with pytest.raises(ValueError, match="question_template"):
        LoadArgs(
            num_proc=1,
            question_cols=["q"],
            question_template="Q",
            answer_cols=["a"],
        )


def test_runtime_args_save_and_upload_behaviors(tmp_path: Path) -> None:
    runtime = RuntimeArgs(
        override_src=None,
        show_first_n=1,
        save_dir=tmp_path / "artifacts",
        save_parquet=False,
        hf=True,
        hf_repo="repo",
        hf_subset="subset",
        hf_private=True,
        dry_run=False,
    )
    dataset = pytest.Mock() if hasattr(pytest, "Mock") else None
    if dataset is None:
        from unittest.mock import Mock

        dataset = Mock()

    runtime.save(dataset, runtime.save_dir / "train")
    dataset.save_to_disk.assert_called_once_with(runtime.save_dir / "train")

    runtime.upload(dataset, split="train")
    dataset.push_to_hub.assert_called_once_with(
        repo_id="repo",
        config_name="subset",
        split="train",
        private=True,
    )

    dry_runtime = RuntimeArgs(
        override_src=None,
        show_first_n=0,
        save_dir=tmp_path / "dry",
        save_parquet=True,
        hf=True,
        hf_repo="repo",
        hf_subset="subset",
        hf_private=False,
        dry_run=True,
    )
    dry_dataset = Dataset.from_list([{"question": "Q", "answer": "A"}])
    dry_runtime.save(dry_dataset, dry_runtime.save_dir / "train.parquet")
    dry_runtime.upload(dry_dataset, split="train")
