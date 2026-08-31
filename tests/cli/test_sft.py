"""CLI tests for the sft target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
INFO     Dataset({
             features: ['images', 'messages', 'id', 'extra_info'],
             num_rows: 10
         })
INFO     {'images': [], 'messages': [{'role': 'user', 'content': 'What is 0+0?'}, {'role': 'assistant', 'content':
         '0'}], 'id': 'qa.jsonl/00000000', 'extra_info': ''}
💾	About to save to disk -> PosixPath('out/sft/auto/train')
💾	Skipping (non-interactive mode)
☁️	About to upload to HF -> 'auto' (subset='default', split='train', private=True)
☁️	Skipping (non-interactive mode)"""


def test_prep_sft_stdout(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "sft", "auto", "train", qa_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
