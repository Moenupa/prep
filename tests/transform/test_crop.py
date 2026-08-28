from PIL import Image

from prep.transform.crop import crop_black_border


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


class TestCropBlackBorder:
    def test_crops_exact_black_frame(self) -> None:
        assert crop_black_border(_framed()).size == (4, 4)

    def test_crops_near_black_frame(self) -> None:
        assert crop_black_border(_framed(bg=(5, 5, 5), fg=(200, 200, 200))).size == (
            4,
            4,
        )

    def test_background_threshold_boundary(self) -> None:
        assert crop_black_border(_framed(bg=(10, 10, 10))).size == (4, 4)
        assert crop_black_border(_framed(bg=(11, 11, 11))).size == (8, 8)

    def test_preserves_off_center_content(self) -> None:
        img = _framed(size=(10, 6), inner=(3, 2), at=(5, 1), fg=(30, 60, 90))
        cropped = crop_black_border(img)
        assert cropped.size == (3, 2)
        assert cropped.getpixel((0, 0)) == (30, 60, 90)

    def test_returns_all_background_unchanged(self) -> None:
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        assert crop_black_border(img) is img

    def test_keeps_non_black_borders(self) -> None:
        assert crop_black_border(_framed(bg=(128, 128, 128))).size == (8, 8)

    def test_preserves_content_and_mode(self) -> None:
        img = _framed(fg=(10, 20, 30))
        cropped = crop_black_border(img)
        assert cropped.mode == "RGB"
        assert cropped.tobytes() == bytes([10, 20, 30] * 16)

        gray = Image.new("L", (8, 8), 0)
        gray.paste(128, (1, 1, 7, 5))
        assert crop_black_border(gray).mode == "L"

        rgba = _framed().convert("RGBA")
        assert crop_black_border(rgba).mode == "RGBA"
