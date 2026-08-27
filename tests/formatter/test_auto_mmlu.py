import pytest

from prep.formatter.auto_eval import load_eval


@pytest.mark.slow
@pytest.mark.parametrize(
    "path, split, expected",
    [
        (
            "cais/mmlu@abstract_algebra",
            "test",
            {
                "id": "mmlu@abstract_algebra/test00000000",
                "images": [],
                "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
                "options": ["0", "4", "2", "6"],
                "answer": 1,
            },
        ),
    ],
)
def test_load_eval(path, split, procargs, expected):
    assert load_eval(path, split, procargs)[0] == expected
