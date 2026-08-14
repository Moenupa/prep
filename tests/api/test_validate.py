import pytest

from prep.api.validate import validate_image_tags, validate_openai_format


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
