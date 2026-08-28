import pytest
from PIL import Image

from prep.api.transform import (
    compose_transforms,
    get_transform,
    transform,
    validate_transform_names,
)


@transform("_test_flip_h")
def _flip_h(img: Image.Image) -> Image.Image:
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


@transform("_test_rotate90")
def _rotate90(img: Image.Image) -> Image.Image:
    return img.transpose(Image.Transpose.ROTATE_90)


def _framed(
    size: tuple[int, int] = (8, 8),
    inner: tuple[int, int] = (4, 4),
    at: tuple[int, int] = (2, 2),
    bg: tuple[int, int, int] = (0, 0, 0),
    fg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    img = Image.new("RGB", size, bg)
    img.paste(Image.new("RGB", inner, fg), at)
    return img


class TestTransformRegistry:
    def test_get_transform_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="crop_black_border"):
            get_transform("no_such_transform")

    def test_validate_transform_names(self) -> None:
        validate_transform_names([])
        validate_transform_names(["crop_black_border"])
        with pytest.raises(KeyError, match="no_such_transform"):
            validate_transform_names(["crop_black_border", "no_such_transform"])

    def test_register_transform_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            transform("_test_flip_h")(lambda img: img)


class TestComposeTransforms:
    def test_compose_empty_is_identity(self) -> None:
        img = _framed()
        assert compose_transforms([])(img) is img

    def test_compose_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="no_such_transform"):
            compose_transforms(["no_such_transform"])

    def test_compose_applies_in_list_order(self) -> None:
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (255, 255, 255))
        img.putpixel((1, 0), (255, 0, 0))

        composed = compose_transforms(["_test_flip_h", "_test_rotate90"])
        expected = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
            Image.Transpose.ROTATE_90
        )
        assert composed(img).tobytes() == expected.tobytes()

        reversed_order = compose_transforms(["_test_rotate90", "_test_flip_h"])
        assert reversed_order(img).tobytes() != composed(img).tobytes()
