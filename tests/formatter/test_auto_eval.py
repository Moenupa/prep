"""Unit tests for auto_eval formatter without requiring Hugging Face datasets."""

from prep.formatter.auto_eval import auto_eval


class TestAutoEvalFunction:
    """Tests for the auto_eval formatting function."""

    def test_auto_eval_basic_multiple_choice(self):
        """Test eval formatting with multiple choice question."""
        example = {
            "question": "What is the capital of France?",
            "A": "London",
            "B": "Paris",
            "C": "Berlin",
            "D": "Madrid",
            "answer": "B",
        }

        result = auto_eval(
            example,
            0,
            data_name="mmlu",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=["A", "B", "C", "D"],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["id"] == "mmlu/test00000000"
        assert result["images"] == []
        assert result["question"] == "What is the capital of France?"
        # Options should be extracted separately
        assert len(result["options"]) == 4
        assert result["answer"] == "B"

    def test_auto_eval_removes_options_from_question_template(self):
        """Test that options placeholder is removed when options exist."""
        example = {
            "question": "Select:",
            "opt1": "Choice A",
            "opt2": "Choice B",
            "answer": "A",
        }

        result = auto_eval(
            example,
            5,
            data_name="test",
            split="val",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=["opt1", "opt2"],
            q_template="{question}{options}",
            a_template="{answer}",
        )

        # Options should NOT appear in the question text
        assert "{options}" not in result["question"]
        assert result["options"] == ["Choice A", "Choice B"]

    def test_auto_eval_text_only_no_options(self):
        """Test eval formatting without options field."""
        example = {
            "question": "Compute the integral.",
            "answer": "\\boxed{42}",
        }

        result = auto_eval(
            example,
            10,
            data_name="math",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["options"] is None or result["options"] == []
        assert result["question"] == "Compute the integral."

    def test_auto_eval_removes_boxed_from_answer(self):
        r"""Test that \\boxed{} wrapper is removed from answer."""
        example = {
            "question": "Solve: x^2 = 4",
            "answer": "\\boxed{2}",
        }

        result = auto_eval(
            example,
            0,
            data_name="math",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["answer"] == "2"

    def test_auto_eval_custom_templates(self):
        """Test eval formatting with custom templates."""
        example = {
            "q_text": "Physics problem here",
            "a_text": "9.8 m/s^2",
        }

        result = auto_eval(
            example,
            0,
            data_name="physics",
            split="train",
            q_cols=["q_text"],
            a_cols=["a_text"],
            option_cols=[],
            q_template="[Q] {q_text} [/Q]",
            a_template="[A] {a_text} [/A]",
        )

        assert result["question"] == "[Q] Physics problem here [/Q]"
        assert result["answer"] == "[A] 9.8 m/s^2 [/A]"

    def test_auto_eval_id_format(self):
        """Test that ID follows expected format."""
        example = {
            "question": "Test",
            "answer": "A",
        }

        result = auto_eval(
            example,
            12345,
            data_name="custom_dataset",
            split="validation",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["id"] == "custom_dataset/validation00012345"

    def test_auto_eval_with_empty_option_cols_string(self):
        """Test handling when option_cols is empty list."""
        example = {
            "question": "Open-ended question",
            "answer": "Long explanation...",
        }

        result = auto_eval(
            example,
            0,
            data_name="qa",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
        )

        assert result["question"] == "Open-ended question"
        assert result["answer"] == "Long explanation..."
