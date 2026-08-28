from collections.abc import Iterator
from typing import Any

IMAGE_FIELDS = ("images", "image", "img")


def iter_images(e: dict) -> Iterator[Any]:
    """Iterate over image entries in an example dictionary.

    Supports the same conventions as :func:`extract_images`: ``images``,
    ``image``, ``img``, ``image_1``..``image_n``, and ``image_01``..``image_99``.
    Values may be PIL images, raw ``{"bytes", "path"}`` records, or file paths,
    depending on whether the dataset decodes its image feature.

    Args:
        e: Example dictionary potentially containing image fields.

    Yields:
        Image entries found in the example.
    """
    if "image_1" in e:
        for i in range(1, 100):
            if f"image_{i}" not in e:
                break
            yield e[f"image_{i}"]
        return

    if "image_01" in e:
        for i in range(1, 100):
            if f"image_{i:02d}" not in e:
                break
            yield e[f"image_{i:02d}"]
        return

    for col in IMAGE_FIELDS:
        value = e.get(col)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            yield from value
        else:
            yield value
        return


def extract_images(e: dict) -> list:
    """Extract image paths from an example dictionary.

    Supports multiple naming conventions:
    - Single image: "image"
    - Numbered images: "image_1", "image_2", ... or "image_01", "image_02", ...
    - List: "images"
    - Alternative: "img"

    Args:
        e: Example dictionary potentially containing image fields.

    Returns:
        List of image paths.
    """
    images = []
    if "image" in e:
        images.append(e["image"])
    elif "image_1" in e:
        for i in range(1, 100):
            if f"image_{i}" in e:
                images.append(e[f"image_{i}"])
            else:
                break
    elif "image_01" in e:
        for i in range(1, 100):
            if f"image_{i:02d}" in e:
                images.append(e[f"image_{i:02d}"])
            else:
                break
    elif "images" in e:
        images = e["images"]
    elif "img" in e:
        images.append(e["img"])

    return images
