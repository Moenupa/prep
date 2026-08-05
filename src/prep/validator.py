from datasets import Features, Image, List, Value
from openai._models import validate_type
from openai.types.chat import ChatCompletionMessageParam

verl_mm_features = Features(
    images=List(Image(decode=True)),
    data_source=Value("string"),
    prompt=List(
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ),
    ability=Value("string"),
    reward_model={
        "style": Value("string"),
        "ground_truth": Value("string"),
    },
    extra_info={
        "split": Value("string"),
        "index": Value("string"),
        # feedback, CoT, or hint to guide better answers
        "explanation": Value("string"),
        # any miscellaneous info accepting json.dumps() stuff
        # this is for compatiblity with multiple datasets, supporting any structure
        "misc": Value("string"),
    },
)

sft_mm_features = Features(
    images=List(Image(decode=True)),
    messages=List(
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ),
    id=Value("string"),
    extra_info=Value("string"),
)


def validate_openai_messages(
    messages: list[dict] | list[ChatCompletionMessageParam],
    expected_n_img: int | None = None,
    img_tag: str = "<image>",
) -> bool:
    try:
        validate_type(type_=list[ChatCompletionMessageParam], value=messages)
    except Exception:
        return False

    # skip image tag '<image>' validation
    if expected_n_img is None:
        return True

    n_img_tag = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            n_img_tag += content.count(img_tag)
        elif isinstance(content, list):
            for part in content:
                if not (isinstance(part, dict) and isinstance(part.get("text"), str)):
                    continue
                n_img_tag += part.get("text", "").count(img_tag)

    if n_img_tag != expected_n_img:
        raise ValueError(
            f"Expected {expected_n_img} images, but found {n_img_tag} {img_tag!r} in {messages}"
        )

    return True
