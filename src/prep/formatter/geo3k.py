from datasets import Dataset, load_dataset

from ..constants import NUM_PROC
from ..registry import register_loader


@register_loader("geo3k", "verl", "train", hf_path="hiyouga/geometry3k")
@register_loader("geo3k", "verl", "test", hf_path="hiyouga/geometry3k")
def load(path: str, split: str) -> Dataset:
    d = load_dataset(path, split=split)
    d = d.map(
        lambda e, idx: {
            "images": e["images"],
            "data_source": "geo3k",
            "prompt": [
                {"role": "user", "content": e["problem"]},
            ],
            "ability": "MATH",
            "reward_model": {
                "style": "math",
                "ground_truth": e["answer"],
            },
            "extra_info": {
                "split": split,
                "index": f"{idx:08d}",
                "explanation": e.get("explanation", ""),
                "misc": e.get("misc", ""),
            },
        },
        remove_columns=d.column_names,  # remove original columns
        num_proc=NUM_PROC,  # accelerate with multiprocessing
        with_indices=True,  # add index information to the mapping function
    )
    return d


@register_loader("geo3k", "sft", "train", hf_path="hiyouga/geometry3k")
@register_loader("geo3k", "sft", "test", hf_path="hiyouga/geometry3k")
def load_sft(path: str, split: str) -> Dataset:
    d = load_dataset(path, split=split)
    d = d.map(
        lambda e, idx: {
            "images": e["images"],
            # msg in openai format, which in this case a two-turn conversation
            "messages": [
                {"role": "user", "content": e["problem"]},
                {"role": "assistant", "content": e["answer"]},
            ],
            # ideally this should include 'data_source/index'
            "id": f"geo3k/{idx:08d}",
            # dump a json string here for any miscellaneous info
            "extra_info": e.get("misc", ""),
        },
        remove_columns=d.column_names,
        num_proc=NUM_PROC,
        with_indices=True,
    )
    return d
