import pytest

from prep.validator import validate_openai_messages


def test_validate_openai_messages() -> None:
    messages = [
        {"role": "user", "content": "<image> Solve"},
        {"role": "assistant", "content": "42"},
    ]
    assert validate_openai_messages(messages, expected_n_img=1) is True
    assert validate_openai_messages([{"role": "assistant", "content": 1}]) is False

    with pytest.raises(ValueError, match="Expected 2 images"):
        validate_openai_messages(messages, expected_n_img=2)
