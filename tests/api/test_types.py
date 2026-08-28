import pytest

from prep.api.types import ProcArgs


def test_load_args_validate_required_fields() -> None:
    with pytest.raises(ValueError, match="num_proc"):
        ProcArgs(
            num_proc=0,
            question_cols=["q"],
            question_template="{question}",
            answer_cols=["a"],
        )

    with pytest.raises(ValueError, match="question_cols"):
        ProcArgs(
            num_proc=1,
            question_cols=[],
            question_template="{question}",
            answer_cols=["a"],
        )

    with pytest.raises(ValueError, match="answer_cols"):
        ProcArgs(
            num_proc=1,
            question_cols=["q"],
            question_template="{question}",
            answer_cols=[],
        )


def test_procargs_warns_for_caption_template_and_peeks_first_and_last(caplog) -> None:
    args = ProcArgs(question_template="caption only", show_first_n=1, show_last_n=1)
    args.peek(["first", "last"])
    assert "does not contain" in caplog.text
    assert "first" in caplog.text and "last" in caplog.text
