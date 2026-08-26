import pytest

from prep.api.formatutils import (
    extract_options,
    format_options,
    get_options_from_multi_entry,
    get_options_from_single_entry,
)


def test_get_options_from_single_entry_sorts_dict_keys_for_stable_option_order() -> (
    None
):
    entry = {
        "choices": {
            "b": "Pleural effusion",
            "a": "No acute finding",
            "c": "Pneumothorax",
        }
    }

    assert get_options_from_single_entry(entry, ["choices"]) == [
        "No acute finding",
        "Pleural effusion",
        "Pneumothorax",
    ]


def test_extract_options_prefers_single_field_representation_over_split_columns() -> (
    None
):
    entry = {
        "options": ["Mass", "Nodule"],
        "choice_a": "Ignored A",
        "choice_b": "Ignored B",
    }

    assert extract_options(entry, ["options", "choice_a", "choice_b"]) == [
        "Mass",
        "Nodule",
    ]


def test_get_options_from_multi_entry_respects_declared_column_order_and_filters_non_strings() -> (
    None
):
    entry = {
        "choice_c": "Third",
        "choice_a": "First",
        "choice_b": None,
        "choice_d": 4,
    }

    assert get_options_from_multi_entry(
        entry,
        ["choice_a", "choice_b", "choice_c", "choice_d"],
    ) == ["First", "Third"]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param([], "", id="empty-options-stay-empty"),
        pytest.param(
            ["Alpha", "Beta", "Gamma"],
            "\nA. Alpha\nB. Beta\nC. Gamma",
            id="options-are-labeled-in-sequence",
        ),
    ],
)
def test_format_options_renders_expected_training_text(
    options: list[str],
    expected: str,
) -> None:
    assert format_options(options) == expected


@pytest.mark.parametrize(
    ("entry", "cols", "expected"),
    [
        pytest.param(
            {"question": "Q", "answer": "A"},
            [],
            [],
            id="empty-cols-returns-empty-list",
        ),
        pytest.param(
            {"option_a": "Choice A"},
            ["option_a"],
            ["Choice A"],
            id="single-col-returns-single-option",
        ),
        pytest.param(
            {
                "A": "Option A text",
                "B": "Option B text",
                "C": "Option C text",
                "D": "Option D text",
            },
            ["A", "B", "C", "D"],
            ["Option A text", "Option B text", "Option C text", "Option D text"],
            id="multiple-cols-returns-all-options",
        ),
        pytest.param(
            {
                "d": "Last",
                "a": "First",
                "c": "Third",
                "b": "Second",
            },
            ["a", "b", "c", "d"],
            ["First", "Second", "Third", "Last"],
            id="preserves-option-order",
        ),
    ],
)
def test_extract_options_handles_various_input_shapes(
    entry: dict,
    cols: list[str],
    expected: list,
) -> None:
    assert extract_options(entry, cols) == expected
