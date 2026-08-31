from typing import TYPE_CHECKING

from ..api import transform

if TYPE_CHECKING:
    from PIL import Image


@transform("convert_rgb")
def convert_rgb(img: "Image.Image") -> "Image.Image":
    """Convert an image to RGB mode.

    Args:
        img: Input image of any mode.

    Returns:
        Image converted to RGB mode.
    """
    return img.convert("RGB")
