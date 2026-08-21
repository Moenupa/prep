import importlib
from io import StringIO

import pytest
from datasets import Dataset

import prep.formatter.auto  # noqa: F401
from prep.api.formatter import FormatterPipeline, formatter
from prep.api.types import SFT_FEAT, ProcArgs, RegistrationError

formatter_api = importlib.import_module("prep.api.formatter")


@pytest.fixture
def procargs() -> ProcArgs:
    return ProcArgs(
        num_proc=1,
        question_cols=["question"],
        question_template="Question: {question}",
        answer_cols=["answer"],
        show_first_n=0,
    )


@pytest.mark.parametrize(
    "pipeline_id, target_format, split",
    [
        # duplicated registration
        ("vqa", "sft", "train"),
        ("vqa", "verl", "val"),
        ("_noop", "show", "val"),
        # invalid id
        ("invalid/pipeline", "sft", "train"),
        ("invalid pipeline", "verl", "test"),
        ("invalid?id", "verl", "test"),
        ("invalid@id", "verl", "test"),
    ],
)
def test_formatter_id_check(pipeline_id: str, target_format: str, split: str) -> None:
    with pytest.raises(RegistrationError):

        @formatter(pipeline_id, target_format, split)  # ty: ignore[invalid-argument-type]
        def dummy(): ...


def test_formatter_get_falls_back_to_vqa_for_unknown_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr_buffer = StringIO()
    monkeypatch.setattr(formatter_api, "stderr", stderr_buffer)

    pipeline = FormatterPipeline.get("some-unknown-pipeline", "sft", "train")

    assert pipeline.id_ == "vqa"
    assert pipeline.target_format == "sft"
    assert "Fallback to the general pipeline 'vqa'" in stderr_buffer.getvalue()


def test_formatter_load_uses_override_source_and_casts_sft_rows(
    procargs: ProcArgs,
) -> None:
    seen: dict[str, object] = {}
    test_sample = {
        "images": [],
        "messages": [
            {
                "role": "user",
                "content": "Question: What does the chest X-ray show?",
            },
            {
                "role": "assistant",
                "content": "No acute cardiopulmonary process.",
            },
        ],
        "id": "Mock/00000001",
        "extra_info": "source=mock note",
    }

    def load_mock_dataset(path: str, split: str, args: ProcArgs) -> Dataset:
        seen["path"] = path
        seen["split"] = split
        seen["question_template"] = args.question_template
        return Dataset.from_list([test_sample])

    pipeline = FormatterPipeline(
        id_="demo",
        target_format="sft",
        split="train",
        load_fn=load_mock_dataset,
        default_src="unused-default",
    )

    dataset = pipeline.load("examples/demo", procargs)

    assert seen == {
        "path": "examples/demo",
        "split": "train",
        "question_template": "Question: {question}",
    }
    assert dataset.features == SFT_FEAT
    assert dataset[0] == test_sample
