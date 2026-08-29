"""CLI tests for the verl target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
INFO     Dataset({
             features: ['images', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'],
             num_rows: 10
         })
INFO     {'images': [], 'data_source': 'qa.jsonl', 'prompt': [{'role': 'user', 'content': 'What is 0+0?\\nPlease reason
         step by step, and put your final answer within \\\\boxed{}.'}], 'ability': 'math', 'reward_model': {'style':
         'rule', 'ground_truth': '0'}, 'extra_info': {'split': 'train', 'index': '00000000', 'explanation': '', 'misc':
         ''}}
💾	About to save to disk -> PosixPath('out/verl/auto/train')
💾	Skipping (non-interactive mode)
☁️	About to upload to HF -> 'auto' (subset='default', split='train', private=True)
☁️	Skipping (non-interactive mode)"""


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
