from collections.abc import Callable, Mapping, Sequence
from io import BytesIO

from PIL import Image

type ImageTransform = Callable[[Image.Image], Image.Image]

_TRANSFORMS: dict[str, ImageTransform] = {}


def transform(name: str) -> Callable[[ImageTransform], ImageTransform]:
    """Register an image-to-image transform under a canonical name.

    Args:
        name: Canonical name of the transform.

    Returns:
        A decorator that registers the given transform function.

    Raises:
        ValueError: If a transform with ``name`` is already registered.
    """

    def decorator(fn: ImageTransform) -> ImageTransform:
        if name in _TRANSFORMS:
            raise ValueError(f"Transform {name!r} already registered")
        _TRANSFORMS[name] = fn
        return fn

    return decorator


def get_transforms() -> Mapping[str, ImageTransform]:
    """Return the registry of known image transforms."""
    return _TRANSFORMS


def get_transform(name: str) -> ImageTransform:
    """Return the image transform registered under ``name``.

    Args:
        name: Name of a registered transform.

    Returns:
        The registered transform function.

    Raises:
        KeyError: If ``name`` is not a registered transform.
    """
    if name not in _TRANSFORMS:
        raise KeyError(
            f"Unknown transform {name!r}."
            f" Available transforms: {list_transform_names()}"
        )
    return _TRANSFORMS[name]


def list_transform_names() -> list[str]:
    """Return the sorted names of all registered transforms."""
    return sorted(_TRANSFORMS)


def validate_transform_names(names: Sequence[str]) -> None:
    """Validate that every name in ``names`` is a registered transform.

    Args:
        names: Transform names to validate.

    Raises:
        KeyError: If any name is not a registered transform.
    """
    unknown = [name for name in names if name not in _TRANSFORMS]
    if unknown:
        raise KeyError(
            f"Unknown transform(s): {unknown}."
            f" Available transforms: {list_transform_names()}"
        )


def compose_transforms(names: Sequence[str]) -> ImageTransform:
    """Compose registered transforms into one, applied in the given order.

    Args:
        names: Transform names, applied sequentially in list order.

    Returns:
        A single transform applying the named transforms in order.
        An empty list yields the identity transform.

    Raises:
        KeyError: If any name is not a registered transform.
    """
    fns = [get_transform(name) for name in names]

    def composed(img: Image.Image) -> Image.Image:
        for fn in fns:
            img = fn(img)
        return img

    return composed


def apply_image_transform(
    image: Image.Image | dict | None, transform: ImageTransform
) -> Image.Image | dict | None:
    """Apply ``transform`` to a dataset image value, preserving its representation.

    Args:
        image: A PIL image or a dict like ``{"bytes": ..., "path": ...}``
            as produced by ``datasets.Image`` features.
        transform: Image-to-image callable to apply.

    Returns:
        The transformed image in the same representation as the input.

    Raises:
        TypeError: If the image representation is not transformable.
    """
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return transform(image)
    if isinstance(image, dict):
        img = _open_image_dict(image)
        out = BytesIO()
        transformed = transform(img)
        transformed.save(out, format=transformed.format or "PNG")
        return {"path": None, "bytes": out.getvalue()}
    raise TypeError(f"Cannot transform image of type {type(image)!r}: {image!r}")


def _open_image_dict(image: dict) -> Image.Image:
    raw = image.get("bytes")
    if raw is not None:
        return Image.open(BytesIO(raw))
    path = image.get("path")
    if path:
        return Image.open(path)
    raise TypeError(f"Cannot open image dict without bytes or path: {image!r}")
