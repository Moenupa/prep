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
