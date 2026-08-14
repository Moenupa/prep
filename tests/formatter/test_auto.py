import pytest

from prep.formatter import auto


def test_auto_parse_images_and_qa() -> None:
    assert auto._parse_images({"image": "a"}) == ["a"]
    assert auto._parse_images({"image_1": "a", "image_2": "b"}) == ["a", "b"]
    assert auto._parse_images({"images": ["a"]}) == ["a"]
    assert auto._parse_images({}) == []

    question, answer = auto._parse_qa(
        {"question": "What?", "answer": "42"},
        q_cols=["question"],
        a_cols=["answer"],
        q_template="Q: {question}",
        n_img_tags=2,
    )
    assert question == "<image><image>Q: What?"
    assert answer == "42"


def test_auto_parse_qa_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="Invalid answer"):
        auto._parse_qa(
            {},
            q_cols=["question"],
            a_cols=["answer"],
            q_template="{question}",
            n_img_tags=0,
        )


def test_auto_sft_does_not_double_apply_template() -> None:
    row = auto.auto_sft(
        {"question": "Solve", "answer": "42", "images": ["img"]},
        7,
        data_name="demo",
        q_cols=["question"],
        a_cols=["answer"],
        q_template="Q: {question}",
    )

    assert row["messages"][0]["content"] == "<image>Q: Solve"
    assert row["id"] == "demo/00000007"
