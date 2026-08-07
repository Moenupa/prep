from typing import TYPE_CHECKING

from ..args import LoadArgs
from ..constants import first_value
from ..load import adaptive_load_dataset
from ..registry import register_loader

if TYPE_CHECKING:
    from datasets import Dataset


def _parse_images(e: dict) -> list:
    if "image" in e:
        images = [e["image"]]
    elif "image_1" in e:
        images = [e[f"image_{i}"] for i in range(1, 100) if f"image_{i}" in e]
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

    if question is None or answer is None:
        raise ValueError(f"Missing question or answer in example: {e}")
    if "<image>" not in question:
        question = "<image>" * n_img_tags + question

    return q_template.format(question=question), answer


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
        "id": f"{data_name}/{idx:08d}",
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
def load_sft(path: str, split: str, loadargs: LoadArgs) -> "Dataset":
    d = adaptive_load_dataset(path, split={"val": "validation"}.get(split, split))
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
def load_verl(path: str, split: str, loadargs: LoadArgs) -> "Dataset":
    d = adaptive_load_dataset(path, split={"val": "validation"}.get(split, split))
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
