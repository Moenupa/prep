from ..api import ProcArgs, adaptive_load_dataset, formatter
from ..api.formatutils import extract_images, extract_qa


def auto_sft(
    e: dict,
    idx: int,
    *,
    data_name: str,
    q_cols: list[str],
    a_cols: list[str],
    option_cols: list[str],
    q_template: str,
    a_template: str,
) -> dict:
    images = extract_images(e)
    question, answer = extract_qa(
        e, q_cols, a_cols, option_cols, q_template, a_template, n_img_tags=len(images)
    )

    return {
        "images": images,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "id": f"{data_name}/{idx:08d}",
        "extra_info": e.get("extra_info", ""),
    }


@formatter("auto", "sft", "train", default_src=None)
@formatter("auto", "sft", "val", default_src=None)
@formatter("auto", "sft", "test", default_src=None)
def load_sft(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    args.peek(d, level=10)
    d = d.map(
        auto_sft,
        fn_kwargs=dict(
            data_name=path.split("/")[-1],
            q_cols=args.question_cols,
            a_cols=args.answer_cols,
            option_cols=args.option_cols,
            q_template=args.question_template,
            a_template=args.answer_template,
        ),
        remove_columns=d.column_names,
        num_proc=args.num_proc,
        with_indices=True,
    )
    return d
