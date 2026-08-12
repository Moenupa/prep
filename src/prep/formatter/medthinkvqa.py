from datasets import load_dataset

from ..args import DatasetPrepStream, LoadArgs
from ..registry import formatter
from .auto import auto_sft, auto_verl

_NAME = "MedThinkVQA"
_SRC = "CAIR-M3LLM/MedThinkVQA"


@formatter(_NAME, "verl", "train", default_src=_SRC)
@formatter(_NAME, "verl", "test", default_src=_SRC)
def load(path: str, split: str, loadargs: LoadArgs) -> DatasetPrepStream:
    d = load_dataset(path, split=split)
    # yield d
    d = d.map(
        auto_verl,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "split": split,
            "q_cols": ["CLINICAL_HISTORY"],
            "a_cols": ["correct_answer"],
            "q_template": "{CLINICAL_HISTORY}{im_tags}"
            "\nBased on ALL provided images together with the textual context"
            ", select the single best diagnosis from the options.{options}"
            "\nPlease reason step by step, and put your final answer within \\boxed{{}}",
        },
        remove_columns=d.column_names,  # remove original columns
        num_proc=loadargs.num_proc,  # accelerate with multiprocessing
        with_indices=True,  # add index information to the mapping function
    )
    return d


@formatter(_NAME, "sft", "train", default_src=_SRC)
@formatter(_NAME, "sft", "test", default_src=_SRC)
def load_caption(path: str, split: str, loadargs: LoadArgs) -> DatasetPrepStream:
    d = load_dataset(path, split=split)

    def expand_batch(batch: dict) -> dict:
        all_images = []
        all_captions = []
        all_ids = []

        # each row has multiple images and captions
        for caseid, images, captions, im_ids in zip(
            batch["caseid"],
            batch["images"],
            batch["image_captions"],
            batch["image_ids"],
        ):
            # pair these two up
            assert len(images) == len(captions) == len(im_ids)

            for image, caption, imid in zip(images, captions, im_ids):
                all_images.append(image)
                all_captions.append(caption)
                all_ids.append(f"{caseid}/{imid}")

        return {"image": all_images, "caption": all_captions, "id": all_ids}

    d = d.map(
        expand_batch,
        batched=True,
        batch_size=100,
        num_proc=loadargs.num_proc,
        remove_columns=d.column_names,
    )
    d = d.map(
        auto_sft,
        fn_kwargs={
            "data_name": path.split("/")[-1],
            "q_cols": [],
            "a_cols": ["caption"],
            "q_template": "",
        },
        remove_columns=d.column_names,
        num_proc=loadargs.num_proc,
        with_indices=True,
    )

    return d
