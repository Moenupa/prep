from typing import TYPE_CHECKING

from ..api import transform

if TYPE_CHECKING:
    from PIL import Image


@transform("crop_black_border")
def crop_black_border(img: "Image.Image") -> "Image.Image":
    """Crop away borders of black or near-black pixels surrounding an image.

    A pixel is treated as background when all of its channels are at or
    below 10. Returns the original image unchanged if the whole image is
    background (e.g. an entirely black image).

    Args:
        img: Input image of any mode.

    Returns:
        Image with black borders removed, same mode as the input.
    """
    mask = img.convert("RGB").point(lambda v: 0 if v <= 10 else 255)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    return img.crop(bbox)


@transform("crop_black_columns")
def crop_black_columns(
    image: "Image.Image", brightness_threshold: int = 35, column_threshold: float = 0.08
) -> "Image.Image":
    import numpy as np

    arr = np.asarray(image.convert("RGB"))

    # A column is kept only when a substantial portion is brighter than black.
    # This rejects sparse white text/icons in otherwise black columns.
    brightness = arr.max(axis=2)
    keep = (brightness > brightness_threshold).mean(axis=0) >= column_threshold

    # Find the longest consecutive run of columns to keep.
    padded = np.pad(keep.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    if starts.size == 0:
        return image

    longest = np.argmax(ends - starts)
    left, right = int(starts[longest]), int(ends[longest])

    return image.crop((left, 0, right, image.height))
