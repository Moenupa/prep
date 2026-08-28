from collections.abc import Generator

from openai._models import validate_type
from openai.types.chat import ChatCompletionMessageParam

from ..constants import FORMATTING_PATTERN, IMAGE_TAG

__all__ = [
    "count_img_tags",
    "validate_image_tags",
    "validate_openai_format",
]


def validate_openai_format(
    messages: list[dict] | list[ChatCompletionMessageParam],
) -> None:
    validate_type(type_=list[ChatCompletionMessageParam], value=messages)


def iter_user_content(
    messages: list[dict] | list[ChatCompletionMessageParam],
) -> Generator[str, None, None]:
    for msg in messages:
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if not (isinstance(part, dict) and isinstance(part.get("text"), str)):
                    continue
                yield part.get("text", "")


def validate_image_tags(
    messages: list[dict] | list[ChatCompletionMessageParam],
    expected_n_img: int | None = None,
) -> None:
    # skip image tag '<image>' validation
    if expected_n_img is None:
        return

    n_img_tag = 0
    for each_user_content in iter_user_content(messages):
        n_img_tag += count_img_tags(each_user_content)

    if expected_n_img is not None and n_img_tag != expected_n_img:
        raise ValueError(
            f"Expected {expected_n_img} images, but found {n_img_tag} {IMAGE_TAG!r} in {messages}"
        )

    return


def validate_answer_formatting(
    messages: list[dict] | list[ChatCompletionMessageParam],
) -> None:
    # if any user msg contains formatting hints, we consider it valid
    for each_user_content in iter_user_content(messages):
        if FORMATTING_PATTERN.search(each_user_content):
            return

    raise SyntaxWarning(
        f"No formatting hints. Is this expected? (detected by regex {FORMATTING_PATTERN.pattern!r})"
    )


def count_img_tags(text: str) -> int:
    if "<image 1>" in text or "<image 01>" in text:
        # support <image 01> and <image 1> tags, 1-99 tags
        # if it goes beyond that, use <image> instead.
        # do not use .count() because it measures multiple occurrences of the same tag
        return sum(
            (f"<image {i:02d}>" in text) or (f"<image {i}>" in text)
            for i in range(1, 100)
        )

    return text.count(IMAGE_TAG)
