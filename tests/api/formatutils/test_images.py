import pytest

from prep.api.formatutils import extract_images


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
