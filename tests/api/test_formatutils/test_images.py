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
    ],
)
def test_extract_images_handles_priority_and_sequence_boundaries(
    entry: dict,
    expected: list[str],
) -> None:
    assert extract_images(entry) == expected
