import pytest

from prep.api.formatutils import extract_images, iter_images


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            {"image": "cover.png", "images": ["ignored.png"]},
            ["cover.png"],
            id="single-image-takes-precedence-over-images-list",
        ),
        pytest.param(
            {
                "image_1": "scan-1.png",
                "image_2": "scan-2.png",
                "image_4": "scan-4.png",
                "images": ["ignored.png"],
            },
            ["scan-1.png", "scan-2.png"],
            id="numbered-images-stop-at-first-gap",
        ),
        pytest.param(
            {
                "image_01": "slice-01.png",
                "image_02": "slice-02.png",
                "image_04": "slice-04.png",
            },
            ["slice-01.png", "slice-02.png"],
            id="zero-padded-images-stop-at-first-gap",
        ),
        pytest.param(
            {"images": ["frame-2.png", "frame-1.png", "frame-3.png"]},
            ["frame-2.png", "frame-1.png", "frame-3.png"],
            id="images-list-preserves-input-order",
        ),
        pytest.param(
            {"question": "Q", "answer": "A"},
            [],
            id="none-field-returns-empty-list",
        ),
        pytest.param(
            {"images": []},
            [],
            id="empty-list-returns-empty",
        ),
        pytest.param(
            {"images": [{"path": "test.png"}]},
            [{"path": "test.png"}],
            id="single-dict-image",
        ),
        pytest.param(
            {"images": [{"path": "1.png"}, {"path": "2.png"}]},
            [{"path": "1.png"}, {"path": "2.png"}],
            id="list-of-dicts-images",
        ),
        pytest.param(
            {"image": {"path": "single.png"}},
            [{"path": "single.png"}],
            id="alternate-image-field-as-dict",
        ),
        pytest.param(
            {"pixel_values": [[0.1, 0.2, 0.3]]},
            [],
            id="pixel-values-field",
        ),
    ],
)
def test_extract_images_handles_priority_and_sequence_boundaries(
    entry: dict,
    expected: list,
) -> None:
    assert extract_images(entry) == expected


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"image": "a.png"}, ["a.png"], id="single-image"),
        pytest.param({"images": ["a.png", "b.png"]}, ["a.png", "b.png"], id="images"),
        pytest.param({"img": "a.png"}, ["a.png"], id="img"),
        pytest.param(
            {"image_1": "a.png", "image_2": "b.png", "image_4": "skip.png"},
            ["a.png", "b.png"],
            id="numbered-stops-at-gap",
        ),
        pytest.param(
            {"image_01": "a.png", "image_02": "b.png"},
            ["a.png", "b.png"],
            id="zero-padded",
        ),
        pytest.param({"image": None}, [], id="none-value-yields-nothing"),
        pytest.param({"question": "q"}, [], id="no-image-field"),
    ],
)
def test_iter_images_yields_entries_across_conventions(
    entry: dict, expected: list
) -> None:
    assert list(iter_images(entry)) == expected
