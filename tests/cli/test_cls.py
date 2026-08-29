import pytest
from typer.testing import CliRunner

from prep.cli import app

EXPECTED = """\
WARNING  ⚠️      Formatter pipeline not registered for ('cifar10', 'cls', 'train').
WARNING  Fallback to generic pipeline 'auto', which may cause unexpected formatting issues.
INFO     Dataset({
             features: ['id', 'image', 'label', 'extra_info'],
             num_rows: 50000
         })
INFO     {'id': 'cifar10/train00000000', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at
         *>, 'label': 0, 'extra_info': ''}
💾	About to save to disk -> PosixPath('out/cls/cifar10/train')
💾	Skipping (non-interactive mode)
☁️\tAbout to upload to HF -> 'cifar10' (subset='default', split='train', private=True)
☁️\tSkipping (non-interactive mode)"""


@pytest.mark.slow
def test_prep_cls_stdout(runner: CliRunner, assert_stdout):
    result = runner.invoke(
        app,
        [
            "prep",
            "cls",
            "cifar10",
            "train",
            "uoft-cs/cifar10",
            "--head",
            "1",
        ],
    )

    assert_stdout(result, EXPECTED)
