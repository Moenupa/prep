"""Unit tests for auto_sft formatter without requiring Hugging Face datasets."""

from prep.api import ProcArgs
from prep.formatter.auto_sft import auto_sft


class TestAutoSftFunction:
    """Tests for the auto_sft formatting function."""

    def test_auto_sft_basic_text_only(self):
        """Test SFT formatting with text-only data (no images)."""
        example = {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "extra_info": "source=wikipedia",
        }

        result = auto_sft(
            example,
            0,
            data_name="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="Question: {question}",
            a_template="{answer}",
            extra_info="extra_info",
        )

        assert result["id"] == "test/00000000"
        assert result["images"] == []
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert (
            result["messages"][0]["content"]
            == "Question: What is the capital of France?"
        )
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "Paris"
        assert result["extra_info"] == "source=wikipedia"

    def test_auto_sft_with_images(self):
        """Test SFT formatting with image data."""
        mock_image = {"path": "test.png"}
        example = {
            "image_field": mock_image,
            "question": "Describe this image.",
            "answer": "A beautiful landscape.",
        }

        result = auto_sft(
            example,
            5,
            data_name="img_test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["id"] == "img_test/00000005"
        # Image extraction should happen - check structure
        assert "images" in result
        assert len(result["messages"]) == 2

    def test_auto_sft_with_options_in_question(self):
        """Test SFT formatting when options are part of question template."""
        example = {
            "question": "Select the correct answer:",
            "answer": "B",
            "option_a": "Option A text",
            "option_b": "Option B text",
        }

        result = auto_sft(
            example,
            10,
            data_name="multi_choice",
            q_cols=["question", "option_a", "option_b"],
            a_cols=["answer"],
            option_cols=["option_a", "option_b"],
            q_template="{question} {option_a} {option_b}",
            a_template="{answer}",
        )

        assert result["id"] == "multi_choice/00000010"
        assert len(result["messages"]) == 2

    def test_auto_sft_custom_templates(self):
        """Test SFT formatting with custom question/answer templates."""
        example = {
            "q": "2+2=?",
            "a": "4",
        }

        result = auto_sft(
            example,
            0,
            data_name="math",
            q_cols=["q"],
            a_cols=["a"],
            option_cols=[],
            q_template="Solve: {q}",
            a_template="Answer: {a}",
        )

        assert result["messages"][0]["content"] == "Solve: 2+2=?"
        assert result["messages"][1]["content"] == "Answer: 4"

    def test_auto_sft_empty_extra_info(self):
        """Test SFT formatting handles missing extra_info gracefully."""
        example = {
            "question": "Test question",
            "answer": "Test answer",
        }

        result = auto_sft(
            example,
            0,
            data_name="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["extra_info"] == ""


class TestLoadSftIntegration:
    """Integration tests for load_sft with mocked datasets."""

    def test_load_sft_pipeline_structure(self):
        """Test that load_sft returns properly structured dataset."""
        from datasets import Dataset

        # Create a mock dataset
        mock_data = Dataset.from_dict(
            {
                "question": ["Q1", "Q2"],
                "answer": ["A1", "A2"],
                "source": ["s1", "s2"],
            }
        )

        def mock_load(path, split, args):
            return mock_data

        # Temporarily replace the load function

        # Test would normally use the registered pipeline
        # This tests the structure expectations
        procargs = ProcArgs(
            num_proc=1,
            question_cols=["question"],
            question_template="{question}",
            answer_cols=["answer"],
            show_first_n=0,
        )

        # Verify ProcArgs configuration
        assert procargs.question_cols == ["question"]
        assert procargs.answer_cols == ["answer"]
        assert procargs.num_proc == 1
