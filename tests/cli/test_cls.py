import pytest
from typer.testing import CliRunner

from prep.cli import app

EXPECTED = """\
WARNING  Formatter pipeline not registered for ('cifar10', 'cls', 'train').
WARNING  Fallback to generic pipeline 'auto', which may cause unexpected formatting issues.
label Counter({0: 5000, 6: 5000, 2: 5000, 7: 5000, 1: 5000, 4: 5000, 5: 5000, 3: 5000, 8: 5000, 9: 5000})
Dataset({
    features: ['id', 'image', 'label', 'extra_info'],
    num_rows: 50000
})
Sample 0: (images ['out/cls/cifar10/train/000000_00.png'])
{
    'id': 'cifar10/train00000000',
    'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at*>,
    'label': 0,
    'extra_info': ''
}
💾\tAbout to save to disk -> PosixPath('out/cls/cifar10/train')...
💾\tNot saved (pass `--save` to override or `export UI=1` to enable interactive prompts)
☁️\tAbout to upload to HF -> 'cifar10' (subset='default', split='train', private=True)...
☁️\tNot uploaded (pass `--hf` to override or `export UI=1` to enable interactive prompts)"""


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
