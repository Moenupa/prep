import pytest

from prep.api import ProcArgs
from prep.formatter.auto_cls import load_cls


@pytest.mark.slow
@pytest.mark.parametrize(
    "path, split, expected, labels",
    [
        (
            "uoft-cs/cifar10",
            "test",
            {
                "id": "cifar10/test00000000",
                "label": 3,
                "extra_info": "",
            },
            [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
        ),
    ],
)
def test_load_eval(path: str, split, expected: dict, labels: list[str]) -> None:
    first_example = load_cls(
        path,
        split,
        ProcArgs(labels=labels),
    )[0]
    first_example.pop("image")
    assert first_example == expected
