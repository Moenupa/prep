"""CLI tests for the show target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
Dataset({
    features: ['question', 'answer'],
    num_rows: 10
})
Sample 0:
{'question': 'What is 0+0?', 'answer': '0'}
💾\tAbout to save to disk -> PosixPath('out/show/_/train')...
💾\tNot saved (pass `--save` to override or `export UI=1` to enable interactive prompts)
☁️\tAbout to upload to HF -> '_' (subset='default', split='train', private=True)...
☁️\tNot uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"""


def test_prep_show_stdout(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "show", "_", "train", qa_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
