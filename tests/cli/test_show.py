"""CLI tests for the show target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
INFO     Dataset({
             features: ['question', 'answer'],
             num_rows: 10
         })
INFO     {'question': 'What is 0+0?', 'answer': '0'}
💾	About to save to disk -> PosixPath('out/show/_/train')
💾	Skipping (non-interactive mode)
☁️	About to upload to HF -> '_' (subset='default', split='train', private=True)
☁️	Skipping (non-interactive mode)"""


def test_prep_show_stdout(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "show", "_", "train", qa_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
