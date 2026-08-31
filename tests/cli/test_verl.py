"""CLI tests for the verl target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
Dataset({
    features: ['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'],
    num_rows: 10
})
Sample 0:
{
    'images': [],
    'data_source': 'qa.jsonl',
    'prompt': [
        {
            'role': 'user',
            'content': 'What is 0+0?\\nPlease reason step by step, and put your final answer within \\\\boxed{}.'
        }
    ],
    'ability': 'math',
    'reward_model': {'style': 'rule', 'ground_truth': '0'},
    'extra_info': {'split': 'train', 'index': '00000000', 'explanation': '', 'misc': ''}
}
💾\tAbout to save to disk -> PosixPath('out/verl/auto/train')...
💾\tNot saved (pass `--save` to override or `export UI=1` to enable interactive prompts)
☁️\tAbout to upload to HF -> 'auto' (subset='default', split='train', private=True)...
☁️\tNot uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"""


def test_prep_verl_stdout(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app,
        [
            "prep",
            "verl",
            "auto",
            "train",
            qa_jsonl,
            "--head",
            "1",
            "--nproc",
            "1",
            "--q-template",
            "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        ],
    )

    assert_stdout(result, EXPECTED)
