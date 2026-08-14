from openai._models import validate_type
from openai.types.chat import ChatCompletionMessageParam

from ..constants import IMAGE_TAG

__all__ = [
    "count_img_tags",
    "validate_image_tags",
    "validate_openai_format",
]


def validate_openai_format(
    messages: list[dict] | list[ChatCompletionMessageParam],
) -> None:
    validate_type(type_=list[ChatCompletionMessageParam], value=messages)


def validate_image_tags(
    messages: list[dict] | list[ChatCompletionMessageParam],
    expected_n_img: int | None = None,
) -> None:
    # skip image tag '<image>' validation
    if expected_n_img is None:
        return

    n_img_tag = 0
    for msg in messages:
        if msg.get("role") not in ["user"]:
            continue

        content = msg.get("content")
        if isinstance(content, str):
            n_img_tag += count_img_tags(content)
        elif isinstance(content, list):
            for part in content:
                if not (isinstance(part, dict) and isinstance(part.get("text"), str)):
                    continue
                n_img_tag += count_img_tags(part.get("text", ""))

    if expected_n_img is not None and n_img_tag != expected_n_img:
        raise ValueError(
            f"Expected {expected_n_img} images, but found {n_img_tag} {IMAGE_TAG!r} in {messages}"
        )

    return


def count_img_tags(text: str) -> int:
    if "<image 1>" in text:
        # support <image 01> and <image 1> tags, 1-99 tags
        # if it goes beyond that, use <image> instead.
        # do not use .count() because it measures multiple occurrences of the same tag
        return sum(
            (f"<image {i:02d}>" in text) or (f"<image {i}>" in text)
            for i in range(1, 100)
        )

    return text.count(IMAGE_TAG)
