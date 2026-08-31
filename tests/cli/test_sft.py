"""CLI tests for the sft target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
INFO     Dataset({
             features: ['images', 'messages', 'id', 'extra_info'],
             num_rows: 10
         })
INFO     {'images': [], 'messages': [{'role': 'user', 'content': 'What is 0+0?'}, {'role': 'assistant', 'content':
         '0'}], 'id': 'qa.jsonl/00000000', 'extra_info': ''}
💾\tAbout to save to disk -> PosixPath('out/sft/auto/train')...
💾\tNot saved (pass `--save` to override or `export UI=1` to enable interactive prompts)
☁️\tAbout to upload to HF -> 'auto' (subset='default', split='train', private=True)...
☁️\tNot uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"""


def test_prep_sft_stdout(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "sft", "auto", "train", qa_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
