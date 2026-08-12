import pytest

from prep.args import LoadArgs


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
