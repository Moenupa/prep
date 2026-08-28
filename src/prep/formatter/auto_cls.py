from ..api import (
    ProcArgs,
    adaptive_load_dataset,
    apply_image_transform,
    compose_transforms,
    formatter,
)
from ..api.formatutils import extract_images, extract_label


def auto_cls(
    e: dict,
    idx: int,
    *,
    data_name: str,
    split: str,
    a_cols: list[str],
    transforms: list[str],
):
    images = extract_images(e)
    if len(images) != 1:
        raise ValueError(f"Expected 1 image, got {len(images)} in example: {e}")
    label = extract_label(e, a_cols)

    image = images[0]
    if transforms:
        image = apply_image_transform(image, compose_transforms(transforms))

    return {
        "id": f"{data_name}/{split}{idx:08d}",
        "image": image,
        "label": label,
        "extra_info": e.get("extra_info", ""),
    }


@formatter("auto", "cls", "train", default_src=None)
@formatter("auto", "cls", "val", default_src=None)
@formatter("auto", "cls", "test", default_src=None)
def load_cls(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    args.peek(d, level=10)
    d = d.map(
        auto_cls,
        fn_kwargs=dict(
            data_name=path.split("/")[-1],
            split=split,
            a_cols=args.answer_cols,
            transforms=args.transforms,
        ),
        remove_columns=d.column_names,
        num_proc=args.num_proc,
        with_indices=True,
    )
    return d
