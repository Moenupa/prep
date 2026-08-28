import pytest

from prep.api.validate import (
    count_img_tags,
    iter_user_content,
    validate_answer_formatting,
    validate_image_tags,
    validate_openai_format,
)


def test_validate_openai_format() -> None:
    messages = [
        {"role": "user", "content": "<image> Solve"},
        {"role": "assistant", "content": "42"},
    ]
    validate_openai_format(messages)

    with pytest.raises(ValueError):
        validate_openai_format([{"role": "assistant", "content": 1}])


def test_validate_image_tags() -> None:
    messages = [
        {"role": "user", "content": "<image> Solve"},
        {"role": "assistant", "content": "42"},
    ]
    validate_image_tags(messages, expected_n_img=1)
    with pytest.raises(ValueError):
        validate_image_tags(messages, expected_n_img=2)


def test_validation_handles_rich_content_and_numbered_image_tags() -> None:
    messages = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": [{"text": "<image 01><image 2> Answer in \\boxed{}"}, {"type": "image"}]},
    ]
    assert list(iter_user_content(messages)) == ["<image 01><image 2> Answer in \\boxed{}"]
    assert count_img_tags(messages[1]["content"][0]["text"]) == 2
    validate_image_tags(messages, expected_n_img=2)
    validate_answer_formatting(messages)


def test_validation_skips_missing_expected_images_and_warns_without_formatting_hint() -> None:
    validate_image_tags([{"role": "user", "content": "plain"}])
    with pytest.raises(SyntaxWarning, match="No formatting hints"):
        validate_answer_formatting([{"role": "user", "content": "plain"}])
