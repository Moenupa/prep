from ..args import DatasetPrepStream, LoadArgs
from ..constants import first_value, get_logger
from ..load import adaptive_load_dataset
from ..registry import register_loader

logger = get_logger(__name__)


def _parse_images(e: dict) -> list:
    if "image" in e:
        images = [e["image"]]
    elif "image_1" in e:
        images = [
            e[f"image_{i}"] for i in range(1, 100) if e.get(f"image_{i}") is not None
        ]
    elif "image_01" in e:
        images = [
            e[f"image_{i:02d}"]
            for i in range(1, 100)
            if e.get(f"image_{i:02d}") is not None
        ]
    elif "images" in e:
        images = e["images"]
    else:
        images = []
    return images


def _parse_qa(
    e: dict, q_cols: list[str], a_cols: list[str], q_template: str, n_img_tags: int
) -> tuple[str, str]:
    # in order of: sft, verl, and other open vqa formats
    if "messages" in e:
        assert len(e["messages"]) == 2, f"Expected 2 messages, got {len(e['messages'])}"
        question = e["messages"][0]["content"]
        answer = e["messages"][1]["content"]
    elif "prompt" in e and "reward_model" in e:
        question = e["prompt"][0]["content"]
        answer = e["reward_model"]["ground_truth"]
    else:
        question = first_value(e, q_cols)
        answer = first_value(e, a_cols)
    question = question or ""

    if isinstance(answer, int):
        answer = chr(65 + answer)  # convert 0-based index to A/B/C/D
    if not isinstance(answer, str):
        raise ValueError(f"Invalid answer {type(answer)} {answer!r} in example: {e}")

    # pad <image> tags if not present in question or template
    if "<image>" not in question and "{im_tags}" not in q_template:
        q_template = "{im_tags}" + q_template

    return q_template.format(
        **(
            e
            | {
                "question": question,
                "im_tags": "<image>" * n_img_tags,
                "options": _parse_options(e),
            }
        )
    ), answer


def _parse_option_list(options: list[str] | dict | None) -> list[str]:
    if options is None:
        return []
    if isinstance(options, dict):
        # return values sorted by key to ensure consistent order
        return [v for _, v in sorted(options.items())]
    if isinstance(options, list):
        return options
    raise ValueError(f"Expect options, got {type(options)}: {options}")


def _parse_options(e: dict) -> str:
    options = _parse_option_list(e.get("options"))
    if not options:
        return ""
    return "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])


def auto_sft(
    e: dict,
    idx: int,
    *,
    data_name: str,
    q_cols: list[str],
    a_cols: list[str],
    q_template: str,
) -> dict:
    images = _parse_images(e)
    question, answer = _parse_qa(e, q_cols, a_cols, q_template, n_img_tags=len(images))

    return {
        "images": images,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "id": f"{data_name}/" + (e.get("id") or f"{idx:08d}"),
        "extra_info": "",
    }


def auto_verl(
    e: dict,
    idx: int,
    *,
    data_name: str,
    split: str,
    q_cols: list[str],
    a_cols: list[str],
    q_template: str,
):
    images = _parse_images(e)
    question, answer = _parse_qa(e, q_cols, a_cols, q_template, n_img_tags=len(images))

    return {
        "images": images,
        "data_source": data_name,
        "prompt": [{"role": "user", "content": question}],
        "ability": "MATH",
        "reward_model": {"style": "math", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": f"{idx:08d}",
            "explanation": e.get("explanation", ""),
            "misc": e.get("misc", ""),
        },
    }


@register_loader("vqa", "sft", "train", default_src=None)
@register_loader("vqa", "sft", "val", default_src=None)
@register_loader("vqa", "sft", "test", default_src=None)
def load_sft(path: str, split: str, loadargs: LoadArgs) -> "DatasetPrepStream":
    d = adaptive_load_dataset(path, split={"val": "validation"}.get(split, split))
    loadargs.peek(d, level=10)
    d = d.map(
        auto_sft,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "q_cols": loadargs.question_cols,
            "a_cols": loadargs.answer_cols,
            "q_template": loadargs.question_template,
        },
        remove_columns=d.column_names,
        num_proc=loadargs.num_proc,
        with_indices=True,
    )
    return d


@register_loader("vqa", "verl", "train", default_src=None)
@register_loader("vqa", "verl", "val", default_src=None)
@register_loader("vqa", "verl", "test", default_src=None)
def load_verl(path: str, split: str, loadargs: LoadArgs) -> "DatasetPrepStream":
    d = adaptive_load_dataset(path, split={"val": "validation"}.get(split, split))
    loadargs.peek(d, level=10)
    d = d.map(
        auto_verl,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "split": split,
            "q_cols": loadargs.question_cols,
            "a_cols": loadargs.answer_cols,
            "q_template": loadargs.question_template,
        },
        remove_columns=d.column_names,
        num_proc=loadargs.num_proc,
        with_indices=True,
    )
    return d
