from ..constants import first_value


def extract_images(e: dict) -> list:
    images = []
    if "image" in e:
        images.append(e["image"])
    elif "image_1" in e:
        for i in range(1, 100):
            if f"image_{i}" in e:
                images.append(e[f"image_{i}"])
            else:
                break
    elif "image_01" in e:
        for i in range(1, 100):
            if f"image_{i:02d}" in e:
                images.append(e[f"image_{i:02d}"])
            else:
                break
    elif "images" in e:
        images = e["images"]

    return images


def extract_qa(
    e: dict,
    q_cols: list[str],
    a_cols: list[str],
    option_cols: list[str],
    q_template: str,
    a_template: str,
    n_img_tags: int,
) -> tuple[str, str]:
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


def get_options_from_single_entry(e: dict, option_cols: list[str]) -> list[str] | None:
    for col in option_cols:
        options = e.get(col)
        if isinstance(options, list):
            return options
        if isinstance(options, dict):
            # return values sorted by key to ensure consistent order
            # e.g. {"A": "text A", "B": "text B", ...} -> ["text A", "text B", ...]
            # or {"b": "text B", "a": "text A", ...} -> ["text A", "text B", ...]
            return [v for _, v in sorted(options.items())]

    return None


def get_options_from_multi_entry(e: dict, option_cols: list[str]) -> list[str]:
    # this will fallback to an empty list if no options are found
    # which is intended for datasets without options
    options: list[str] = []
    for col in option_cols:
        if isinstance(e.get(col), str):
            options.append(e[col])
    return options


def extract_options(e: dict, option_cols: list[str]) -> list[str]:
    # case 1: e["options"] = {"A": "text A", "B": "text B", ...}
    # case 2: e["options"] as ["text A", "text B", ...]
    # case 3: e["option_1"] = "text A", e["option_2"] = "text B", ...
    # case 4: e["choice_a"] = "text A", e["choice_b"] = "text B", ...
    # assume option_cols is a list containing all the above possible key names

    # case 1&2 first, then fallback to case 3&4, which is more likely to have problems
    options = get_options_from_single_entry(
        e, option_cols
    ) or get_options_from_multi_entry(e, option_cols)
    return options


def format_options(options: list[str]) -> str:
    if not options:
        return ""
    return "\n" + (
        "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
    )
