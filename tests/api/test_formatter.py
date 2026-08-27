import importlib

import pytest
from datasets import ClassLabel, Dataset, List, Value

from prep.api.formatter import FormatterPipeline, formatter
from prep.api.types import ProcArgs, RegistrationError

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


def test_formatter_get_falls_back_to_auto_for_unknown_pipeline(
    capsys: pytest.CaptureFixture,
) -> None:
    pipeline = FormatterPipeline.get("some-unknown-pipeline", "sft", "train")

    assert pipeline.id_ == "auto"
    assert pipeline.target_format == "sft"
    stdout_msg: str = capsys.readouterr().out
    assert "fallback" in stdout_msg.lower() and "auto" in stdout_msg


class TestFormatterPipelineValidation:
    """Additional tests for FormatterPipeline validation."""

    def test_get_pipeline_returns_correct_format(self):
        """Test that get returns pipeline with correct format."""
        pipeline = FormatterPipeline.get("auto", "sft", "train")

        assert pipeline.id_ == "auto"
        assert pipeline.target_format == "sft"
        assert pipeline.split == "train"

    @pytest.mark.parametrize(
        "pipeline_id, target_format, split",
        [
            # duplicated registration
            ("auto", "sft", "train"),
            ("auto", "verl", "val"),
            ("_noop", "show", "val"),
            # invalid id
            ("invalid/pipeline", "sft", "train"),
            ("invalid pipeline", "verl", "test"),
            ("invalid?id", "verl", "test"),
            ("invalid@id", "verl", "test"),
        ],
    )
    def test_formatter_id_check(
        self, pipeline_id: str, target_format: str, split: str
    ) -> None:
        with pytest.raises(RegistrationError):

            @formatter(pipeline_id, target_format, split)  # ty: ignore[invalid-argument-type]
            def dummy(): ...


class TestFormatterCastDataset:
    """Tests for the cast_dataset method in FormatterPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.procargs = ProcArgs(
            num_proc=1,
            question_cols=["question"],
            question_template="Question: {question}",
            answer_cols=["answer"],
            show_first_n=0,
        )

    def test_cast_sft_format(self):
        """Test casting to SFT format."""
        dataset = Dataset.from_dict(
            {
                "id": ["test-001"],
                "images": [[]],
                "messages": [[{"role": "user", "content": "Hello"}]],
                "extra_info": ["source=test"],
            }
        )

        result = FormatterPipeline.cast_dataset(dataset, "sft", self.procargs)

        assert "images" in result.features
        assert "messages" in result.features
        assert "id" in result.features
        assert "extra_info" in result.features

        # Check feature types
        assert isinstance(result.features["images"], List)
        assert isinstance(result.features["messages"], List)
        assert isinstance(result.features["id"], Value)
        assert isinstance(result.features["extra_info"], Value)

    def test_cast_verl_format(self):
        """Test casting to verl format."""
        dataset = Dataset.from_dict(
            {
                "images": [[]],
                "data_source": ["test-data"],
                "prompt": [[{"role": "user", "content": "Hello"}]],
                "ability": ["math"],
                "reward_model": [{"style": "rule", "ground_truth": "42"}],
                "extra_info": [
                    {"split": "test", "index": "001", "explanation": "", "misc": ""}
                ],
            }
        )

        result = FormatterPipeline.cast_dataset(dataset, "verl", self.procargs)

        assert "images" in result.features
        assert "data_source" in result.features
        assert "prompt" in result.features
        assert "ability" in result.features
        assert "reward_model" in result.features
        assert "extra_info" in result.features

    def test_cast_eval_format(self):
        """Test casting to eval format."""
        dataset = Dataset.from_dict(
            {
                "id": ["mmlu-test-001"],
                "images": [[]],
                "question": ["Find the degree..."],
                "options": [["0", "4", "2", "6"]],
                "answer": ["4"],
            }
        )

        result = FormatterPipeline.cast_dataset(dataset, "eval", self.procargs)

        assert "id" in result.features
        assert "images" in result.features
        assert "question" in result.features
        assert "options" in result.features
        assert "answer" in result.features

    def test_cast_cls_format_requires_labels(self):
        """Test that classification format requires labels."""
        dataset = Dataset.from_dict(
            {
                "id": ["cifar-001"],
                "image": [None],
                "label": ["airplane"],
                "extra_info": [""],
            }
        )

        args_with_labels = ProcArgs(
            num_proc=1,
            question_cols=["question"],
            question_template="Question: {question}",
            answer_cols=["answer"],
            labels=["airplane", "automobile", "bird", "cat"],
            show_first_n=0,
        )

        result = FormatterPipeline.cast_dataset(dataset, "cls", args_with_labels)

        assert "id" in result.features
        assert "image" in result.features
        assert "label" in result.features
        assert "extra_info" in result.features

        # Check that label is ClassLabel with correct names
        assert isinstance(result.features["label"], ClassLabel)
        assert result.features["label"].names == [
            "airplane",
            "automobile",
            "bird",
            "cat",
        ]

    def test_cast_cls_format_fails_without_labels(self):
        """Test that classification format fails without labels."""
        dataset = Dataset.from_dict(
            {
                "id": ["cifar-001"],
                "image": [None],
                "label": ["airplane"],
                "extra_info": [""],
            }
        )

        args_without_labels = ProcArgs(
            num_proc=1,
            question_cols=["question"],
            question_template="Question: {question}",
            answer_cols=["answer"],
            labels=[],  # No labels provided
            show_first_n=0,
        )

        with pytest.raises(ValueError):
            FormatterPipeline.cast_dataset(dataset, "cls", args_without_labels)
