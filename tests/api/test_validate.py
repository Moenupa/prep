import pytest

from prep.api.validate import (
    validate_answer_formatting,
    validate_image_tags,
    validate_openai_format,
)


def test_validate_openai_format():
    messages = [
        {"role": "user", "content": "<image> Solve"},
        {"role": "assistant", "content": "42"},
    ]
    validate_openai_format(messages)

    with pytest.raises(ValueError):
        validate_openai_format([{"role": "assistant", "content": 1}])


class TestValidateImageTags:
    @pytest.mark.parametrize(
        "messages,expected_n_img",
        [
            (
                [
                    {"role": "user", "content": "<image> Solve"},
                    {"role": "assistant", "content": "42"},
                ],
                1,
            ),
            (
                [
                    {"role": "user", "content": "<image> Solve"},
                    {"role": "assistant", "content": "42"},
                    {"role": "user", "content": "<image> Solve again"},
                ],
                2,
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"text": "<image 01><image 2> Answer in \\boxed{}"},
                            {"type": "image"},
                        ],
                    },
                ],
                2,
            ),
        ],
    )
    def test_validate_image_tags(self, messages, expected_n_img):
        validate_image_tags(messages, expected_n_img=expected_n_img)
        with pytest.raises(ValueError):
            validate_image_tags(messages, expected_n_img=expected_n_img + 1)


def test_validation_skips_missing_expected_images_and_warns_without_formatting_hint():
    validate_image_tags([{"role": "user", "content": "plain"}])
    with pytest.raises(SyntaxWarning, match="No formatting hints"):
        validate_answer_formatting([{"role": "user", "content": "plain"}])
