from ..api import ProcArgs, adaptive_load_dataset, formatter
from ..api.formatutils import extract_images, extract_qa


def auto_verl(
    e: dict,
    idx: int,
    *,
    data_name: str,
    split: str,
    q_cols: list[str],
    a_cols: list[str],
    option_cols: list[str],
    q_template: str,
    a_template: str,
    ability: str,
    style: str,
    extra_info: str | None = None,
):
    images = extract_images(e)
    question, answer = extract_qa(
        e, q_cols, a_cols, option_cols, q_template, a_template, n_img_tags=len(images)
    )
    # remove \boxed{} wrapper if present, since verl expects raw answer
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{") : -1]  # remove \boxed{} wrapper

    # see https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    return {
        "images": images,
        "data_source": data_name,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": style, "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": f"{idx:08d}",
            "explanation": e.get("explanation", ""),
            "misc": e.get(extra_info, extra_info) or "",
        },
    }


@formatter("auto", "verl", "train", default_src=None)
@formatter("auto", "verl", "val", default_src=None)
@formatter("auto", "verl", "test", default_src=None)
def load_verl(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    args.peek(d, level=10)
    d = d.map(
        auto_verl,
        fn_kwargs=dict(
            data_name=path.split("/")[-1],
            split=split,
            q_cols=args.question_cols,
            a_cols=args.answer_cols,
            option_cols=args.option_cols,
            q_template=args.question_template,
            a_template=args.answer_template,
            ability=args.verl_ability,
            style=args.verl_style,
            extra_info=args.extra_info,
        ),
        remove_columns=d.column_names,
        num_proc=args.num_proc,
        with_indices=True,
    )
    return d
