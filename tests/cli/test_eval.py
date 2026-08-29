"""CLI tests for the eval target format (exact stdout comparison)."""

from prep.cli import app

EXPECTED = """\
INFO     Dataset({
             features: ['id', 'images', 'question', 'options', 'answer'],
             num_rows: 10
         })
INFO     {'id': 'mc.jsonl/test00000000', 'images': [], 'question': 'Pick the even number among the choices (set 0).',
         'options': ['1', '2', '3'], 'answer': 'B'}
💾	About to save to disk -> PosixPath('out/eval/auto/test')
💾	Skipping (non-interactive mode)
☁️	About to upload to HF -> 'auto' (subset='default', split='test', private=True)
☁️	Skipping (non-interactive mode)"""


def test_prep_eval_stdout(runner, mc_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "eval", "auto", "test", mc_jsonl, "--head", "1", "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED)
