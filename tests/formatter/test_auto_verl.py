"""Unit tests for auto_verl formatter without requiring Hugging Face datasets."""

from prep.formatter.auto_verl import auto_verl


class TestAutoVerlFunction:
    """Tests for the auto_verl formatting function."""

    def test_auto_verl_basic(self):
        """Test verl formatting with basic text data."""
        example = {
            "question": "What is 2+2?",
            "answer": "4",
            "extra_info": "source=test",
        }

        result = auto_verl(
            example,
            0,
            data_name="test",
            split="train",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
            ability="math",
            style="rule",
        )

        assert result["data_source"] == "test"
        assert result["ability"] == "math"
        assert result["images"] == []
        assert len(result["prompt"]) == 1
        assert result["prompt"][0]["role"] == "user"
        assert result["prompt"][0]["content"] == "What is 2+2?"
        assert result["reward_model"]["style"] == "rule"
        assert result["reward_model"]["ground_truth"] == "4"
        assert result["extra_info"]["split"] == "train"
        assert result["extra_info"]["index"] == "00000000"

    def test_auto_verl_removes_boxed_wrapper(self):
        r"""Test that \\boxed{} wrapper is removed from answers."""
        example = {
            "question": "Solve for x",
            "answer": "\\boxed{42}",
        }

        result = auto_verl(
            example,
            0,
            data_name="math_test",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
            ability="math",
            style="rule",
        )

        # verl expects raw answer without boxed wrapper
        assert result["reward_model"]["ground_truth"] == "42"

    def test_auto_verl_with_custom_ability_style(self):
        """Test verl formatting with custom ability and style."""
        example = {
            "question": "Explain photosynthesis.",
            "answer": "Process by which plants convert light...",
        }

        result = auto_verl(
            example,
            1,
            data_name="science",
            split="val",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
            ability="science",
            style="llm_judge",
        )

        assert result["ability"] == "science"
        assert result["reward_model"]["style"] == "llm_judge"
        assert result["data_source"] == "science"

    def test_auto_verl_extra_info_fields(self):
        """Test that extra_info contains all required fields."""
        example = {
            "question": "Question text",
            "answer": "Answer text",
            "explanation": "Detailed explanation here",
            "extra_info": "misc info",
        }

        result = auto_verl(
            example,
            100,
            data_name="test",
            split="train",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
            ability="math",
            style="rule",
        )

        assert result["extra_info"]["split"] == "train"
        assert result["extra_info"]["index"] == "00000100"
        assert result["extra_info"]["explanation"] == "Detailed explanation here"
        assert result["extra_info"]["misc"] == "misc info"

    def test_auto_verl_empty_explanation_fallback(self):
        """Test verl formatting handles missing explanation gracefully."""
        example = {
            "question": "Q?",
            "answer": "A",
        }

        result = auto_verl(
            example,
            0,
            data_name="test",
            split="test",
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=[],
            q_template="{question}",
            a_template="{answer}",
            ability="math",
            style="rule",
        )

        assert result["extra_info"]["explanation"] == ""
        assert result["extra_info"]["misc"] == ""

    def test_auto_verl_preserves_question_in_prompt(self):
        """Test that question template is properly applied in prompt."""
        example = {
            "math_problem": "Find x where x + 5 = 10",
            "solution": "5",
        }

        result = auto_verl(
            example,
            0,
            data_name="algebra",
            split="train",
            q_cols=["math_problem"],
            a_cols=["solution"],
            option_cols=[],
            q_template="Math Problem: {math_problem}",
            a_template="Solution: {solution}",
            ability="math",
            style="rule",
        )

        assert result["prompt"][0]["content"] == "Math Problem: Find x where x + 5 = 10"
