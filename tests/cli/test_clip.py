"""CLI tests for the clip target format (exact stdout comparison).

No pipelines are registered for 'clip', so the CLI falls back to 'auto',
which is also unregistered for this format, and fails with a KeyError.
"""

from prep.cli import app

EXPECTED = """\
WARNING  ⚠️      Formatter pipeline not registered for ('auto', 'clip', 'train').
WARNING  Fallback to generic pipeline 'auto', which may cause unexpected formatting issues."""


def test_prep_clip_fallback_fails(runner, qa_jsonl, assert_stdout):
    result = runner.invoke(
        app, ["prep", "clip", "auto", "train", qa_jsonl, "--nproc", "1"]
    )

    assert_stdout(result, EXPECTED, exit_code=1)
    assert isinstance(result.exception, KeyError)
