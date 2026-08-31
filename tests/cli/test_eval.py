"""CLI tests for the eval target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
Dataset({
    features: ['id', 'images', 'question', 'options', 'answer'],
    num_rows: 10
})
Sample 0:
{
    'id': 'mc.jsonl/test00000000',
    'images': [],
    'question': 'Pick the even number among the choices (set 0).',
    'options': ['1', '2', '3'],
    'answer': 'B'
}
💾\tAbout to save to disk -> PosixPath('out/eval/auto/test')...
💾\tNot saved (pass `--save` to override or `export UI=1` to enable interactive prompts)
☁️\tAbout to upload to HF -> 'auto' (subset='default', split='test', private=True)...
☁️\tNot uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"""


def test_prep_eval_stdout(runner, mc_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "eval", "auto", "test", mc_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
