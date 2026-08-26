import pytest

from prep.api.formatutils import extract_label


@pytest.mark.parametrize(
    ("entry", "cols", "expected"),
    [
        pytest.param(
            {"label": "cat"},
            ["label"],
            "cat",
            id="direct-column-returns-value",
        ),
        pytest.param(
            {"class": "car"},
            ["class"],
            "car",
            id="alternate-class-name",
        ),
        pytest.param(
            {"category": "dog"},
            ["category"],
            "dog",
            id="alternate-category-name",
        ),
        pytest.param(
            {"target": 5},
            ["target"],
            5,
            id="alternate-target-name",
        ),
        pytest.param(
            {"label": "primary", "class": "secondary"},
            ["label", "class"],
            "primary",
            id="first-matching-col-is-used",
        ),
        pytest.param(
            {"label": 42},
            ["label"],
            42,
            id="numeric-label-value",
        ),
    ],
)
def test_extract_label_handles_various_input_shapes(
    entry: dict,
    cols: list[str],
    expected,
) -> None:
    assert extract_label(entry, cols) == expected
