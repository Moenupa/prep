"""Question-answer extraction utilities."""

from .helpers import first_value
from .options import extract_options, format_options


def extract_qa(
    e: dict,
    q_cols: list[str],
    a_cols: list[str],
    option_cols: list[str],
    q_template: str,
    a_template: str,
    n_img_tags: int,
) -> tuple[str, str]:
    """Extract question and answer from an example dictionary.

    Supports multiple formats:
    - SFT format with "messages" field
    - Conversation format with "conversations" field
    - Prompt/reward model format
    - Generic column-based extraction

    Args:
        e: Example dictionary.
        q_cols: List of possible question column names.
        a_cols: List of possible answer column names.
        option_cols: List of possible option column names.
        q_template: Template for formatting the question.
        a_template: Template for formatting the answer.
        n_img_tags: Number of <image> tags to insert.

    Returns:
        Tuple of (formatted_question, formatted_answer).

    Raises:
        NotImplementedError: If message/conversation count != 2.
        ValueError: If answer is not a string.
    """
    # in order of: sft, verl, and other open vqa formats
    if "messages" in e:
        if len(e["messages"]) != 2:
            raise NotImplementedError(
                f"Expected 2 messages, got {len(e['messages'])} in example: {e}"
            )
        question = e["messages"][0]["content"]
        answer = e["messages"][1]["content"]
    elif "conversations" in e:
        if len(e["conversations"]) != 2:
            raise NotImplementedError(
                f"Expected 2 conversations, got {len(e['conversations'])} in example: {e}"
            )
        question = e["conversations"][0]["value"]
        answer = e["conversations"][1]["value"]
    elif "prompt" in e and "reward_model" in e:
        question = "\n\n".join(turn["content"] for turn in e["prompt"])
        answer = e["reward_model"]["ground_truth"]
    else:
        # get first non-null value from q_cols and a_cols
        question = first_value(e, q_cols)
        answer = first_value(e, a_cols)
    question = question or ""
    options = extract_options(e, option_cols)

    # in case answers are ClassLabels (0-indexed) instead of strings, convert to A/B/C/D
    if isinstance(answer, int) and options:
        answer = chr(65 + answer)  # convert 0-based index to A/B/C/D
    if not isinstance(answer, str):
        raise ValueError(
            f"Expected answer as str, got {type(answer)} {answer!r} in example: {e}"
        )

    # left-pad <image> tag placeholder to be filled later
    if "<image>" not in question and "{im_tags}" not in q_template:
        q_template = "{im_tags}" + q_template

    return q_template.format(
        **(
            e
            | {
                "question": question,
                "im_tags": "<image>" * n_img_tags,
                "options": format_options(options),
            }
        )
    ), a_template.format(
        **(
            e
            | {
                "answer": answer,
            }
        )
    )
