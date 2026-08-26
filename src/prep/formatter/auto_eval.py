from ..api import ProcArgs, adaptive_load_dataset, formatter
from ..api.formatutils import extract_images, extract_options, extract_qa


def auto_eval(
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
):
    images = extract_images(e)
    options = extract_options(e, option_cols)

    # options should be provided separately in the `options` field,
    # not embedded in the question text
    eval_q_template = q_template.replace("{options}", "") if options else q_template

    question, answer = extract_qa(
        e,
        q_cols,
        a_cols,
        option_cols,
        eval_q_template,
        a_template,
        n_img_tags=len(images),
    )
    # remove \boxed{} wrapper if present, since verl expects raw answer
    if answer.startswith("\\boxed{") and answer.endswith("}"):
        answer = answer[len("\\boxed{") : -1]  # remove \boxed{} wrapper

    # see https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html
    return {
        "id": f"{data_name}/{split}{idx:08d}",
        "images": images,
        "question": question,
        "options": options,
        "answer": answer,
    }


@formatter("auto", "eval", "train", default_src=None)
@formatter("auto", "eval", "val", default_src=None)
@formatter("auto", "eval", "test", default_src=None)
def load_eval(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    args.peek(d, level=10)
    d = d.map(
        auto_eval,
        fn_kwargs=dict(
            data_name=path.split("/")[-1],
            split=split,
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
