from datasets import Dataset, load_dataset

from ..constants import (
    ANSWER_COLS,
    NUM_PROC,
    QUESTION_COLS,
    QUESTION_TEMPLATE,
    first_value,
)
from ..registry import register_loader


def auto_sft(
    e: dict,
    idx: int,
    *,
    data_name: str,
    question_cols: list[str],
    answer_cols: list[str],
    question_template: str,
) -> dict:
    # a generic mapping function for single-turn VQA questions
    # 1. 'question', 'answer', 'image' field
    if "image" in e:
        images = [e["image"]]
    elif "images" in e:
        images = e["images"]
    else:
        images = []

    question = first_value(e, question_cols)
    answer = first_value(e, answer_cols)

    if "messages" in e:
        assert len(e["messages"]) == 2, f"Expected 2 messages, got {len(e['messages'])}"
        question = e["messages"][0]["content"]
        answer = e["messages"][1]["content"]

    if question is None or answer is None:
        raise ValueError(f"Missing question or answer in example {idx}: {e}")

    if "<image>" not in question:
        question = "<image>" * len(images) + question
    return {
        "images": images,
        "messages": [
            {"role": "user", "content": question_template.format(question=question)},
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
    question_cols: list[str],
    answer_cols: list[str],
    question_template: str,
):
    sft = auto_sft(
        e,
        idx,
        data_name=data_name,
        question_cols=question_cols,
        answer_cols=answer_cols,
        question_template=question_template,
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
def load_sft(path: str, split: str) -> Dataset:
    d = load_dataset(path, split={"val": "validation"}.get(split, split))
    d = d.map(
        auto_sft,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "question_cols": QUESTION_COLS,
            "answer_cols": ANSWER_COLS,
            "question_template": QUESTION_TEMPLATE,
        },
        remove_columns=d.column_names,
        num_proc=NUM_PROC,
        with_indices=True,
    )
    return d


@register_loader("vqa", "verl", "train", default_src=None)
@register_loader("vqa", "verl", "val", default_src=None)
@register_loader("vqa", "verl", "test", default_src=None)
def load_verl(path: str, split: str) -> Dataset:
    d = load_dataset(path, split={"val": "validation"}.get(split, split))
    d = d.map(
        auto_verl,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "split": split,
            "question_cols": QUESTION_COLS,
            "answer_cols": ANSWER_COLS,
            "question_template": QUESTION_TEMPLATE,
        },
        remove_columns=d.column_names,
        num_proc=NUM_PROC,
        with_indices=True,
    )
    return d
