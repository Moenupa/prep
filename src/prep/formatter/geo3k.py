from ..api import ProcArgs, adaptive_load_dataset, formatter


@formatter("geo3k", "verl", "train", default_src="hiyouga/geometry3k")
@formatter("geo3k", "verl", "test", default_src="hiyouga/geometry3k")
def load(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    # geo3k has 3 columns: images: list[Image], problem: str, answer: str
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
                "explanation": "",
                "misc": "",
            },
        },
        remove_columns=d.column_names,  # remove original columns
        num_proc=args.num_proc,  # accelerate with multiprocessing
        with_indices=True,  # add index information to the mapping function
    )
    return d


@formatter("geo3k", "sft", "train", default_src="hiyouga/geometry3k")
@formatter("geo3k", "sft", "test", default_src="hiyouga/geometry3k")
def load_sft(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split)
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
            "extra_info": "",
        },
        remove_columns=d.column_names,
        num_proc=args.num_proc,
        with_indices=True,
    )
    return d
