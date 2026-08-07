from datasets import Dataset, load_dataset

from ..args import LoadArgs
from ..constants import (
    first_value,
)
from ..registry import register_loader


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
    e: dict, q_cols: list[str], a_cols: list[str], q_template: str
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
    question, answer = _parse_qa(e, q_cols, a_cols, q_template)

    if "<image>" not in question:
        question = "<image>" * len(images) + question
    return {
        "images": images,
        "messages": [
            {"role": "user", "content": q_template.format(question=question)},
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
    sft = auto_sft(
        e, idx, data_name=data_name, q_cols=q_cols, a_cols=a_cols, q_template=q_template
    )
    verl = {
        "images": sft["images"],
        "data_source": data_name,
        "prompt": sft["messages"][:1],  # only the user message
        "ability": "MATH",
        "reward_model": {
            "style": "math",
            "ground_truth": sft["messages"][1]["content"],
        },
        "extra_info": {
            "split": split,
            "index": f"{idx:08d}",
            "explanation": e.get("explanation", ""),
            "misc": e.get("misc", ""),
        },
    }
    return verl


@register_loader("vqa", "sft", "train", default_src=None)
@register_loader("vqa", "sft", "val", default_src=None)
@register_loader("vqa", "sft", "test", default_src=None)
def load_sft(path: str, split: str, loadargs: LoadArgs) -> Dataset:
    d = load_dataset(path, split={"val": "validation"}.get(split, split))
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
def load_verl(path: str, split: str, loadargs: LoadArgs) -> Dataset:
    d = load_dataset(path, split={"val": "validation"}.get(split, split))
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
