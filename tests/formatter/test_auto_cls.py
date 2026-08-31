"""Unit tests for auto_cls formatter without requiring Hugging Face datasets."""

from io import BytesIO

import pytest
from datasets import Dataset, Value
from datasets import Features as DatasetFeatures
from datasets import Image as DatasetImage
from PIL import Image

from prep.api import FormatterPipeline, ProcArgs
from prep.formatter.auto_cls import auto_cls


def _framed_image() -> Image.Image:
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    img.paste(Image.new("RGB", (4, 4), (255, 255, 255)), (2, 2))
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestAutoClsFunction:
    """Tests for the auto_cls formatting function."""

    def test_auto_cls_basic(self):
        """Test classification formatting with basic data."""
        mock_image = {"path": "cat.png"}
        example = {
            "image": mock_image,
            "label": "cat",
            "extra_info": "source=test",
        }

        result = auto_cls(
            example,
            0,
            data_name="test",
            split="train",
            a_cols=["label"],
            transforms=[],
            extra_info="extra_info",
        )

        assert result["id"] == "test/train00000000"
        assert result["image"] == mock_image
        assert result["label"] == "cat"
        assert result["extra_info"] == "source=test"

    def test_auto_cls_single_image_required(self):
        """Test that auto_cls requires exactly one image."""
        example = {
            "images": [{"path": "img1.png"}, {"path": "img2.png"}],  # Multiple images
            "label": "cat",
        }

        with pytest.raises(ValueError):
            auto_cls(
                example,
                0,
                data_name="test",
                split="train",
                a_cols=["label"],
                transforms=[],
            )

    def test_auto_cls_empty_extra_info(self):
        """Test classification formatting handles missing extra_info."""
        mock_image = {"path": "dog.png"}
        example = {
            "image": mock_image,
            "label": "dog",
        }

        result = auto_cls(
            example, 5, data_name="pets", split="test", a_cols=["label"], transforms=[]
        )

        assert result["extra_info"] == ""

    def test_auto_cls_id_format_with_split(self):
        """Test that ID includes split information."""
        mock_image = {"path": "bird.png"}
        example = {
            "image": mock_image,
            "label": "bird",
        }

        result_train = auto_cls(
            example,
            0,
            data_name="birds",
            split="train",
            a_cols=["label"],
            transforms=[],
        )
        result_val = auto_cls(
            example, 1, data_name="birds", split="val", a_cols=["label"], transforms=[]
        )
        result_test = auto_cls(
            example, 2, data_name="birds", split="test", a_cols=["label"], transforms=[]
        )

        assert result_train["id"] == "birds/train00000000"
        assert result_val["id"] == "birds/val00000001"
        assert result_test["id"] == "birds/test00000002"

    def test_auto_cls_numeric_labels(self):
        """Test classification with numeric label values."""
        mock_image = {"path": "image.jpg"}
        example = {
            "image": mock_image,
            "label": 3,  # Numeric label
        }

        result = auto_cls(
            example,
            10,
            data_name="cifar",
            split="test",
            a_cols=["label"],
            transforms=[],
        )

        assert result["label"] == 3

    def test_auto_cls_custom_answer_column(self):
        """Test classification with custom answer column name."""
        mock_image = {"path": "plane.png"}
        example = {
            "image": mock_image,
            "category": "airplane",
        }

        result = auto_cls(
            example,
            0,
            data_name="cifar10",
            split="test",
            a_cols=["category"],
            transforms=[],
        )

        assert result["label"] == "airplane"


class TestImageExtractionEdgeCases:
    """Tests for edge cases in image extraction for classification."""

    def test_auto_cls_with_none_image(self):
        """Test handling of None image value."""
        example = {
            "image": None,
            "label": "unknown",
        }

        # Should handle None gracefully (depends on extract_images implementation)
        result = auto_cls(
            example, 0, data_name="test", split="train", a_cols=["label"], transforms=[]
        )

        # Image will be None, but structure should be valid
        assert result["id"] == "test/train00000000"
        assert result["label"] == "unknown"

    def test_auto_cls_index_incrementing(self):
        """Test that index is properly incremented across examples."""
        mock_image = {"path": "img.png"}

        results = []
        for i in range(5):
            example = {"image": mock_image, "label": f"label{i}"}
            result = auto_cls(
                example,
                i,
                data_name="seq",
                split="train",
                a_cols=["label"],
                transforms=[],
            )
            results.append(result)

        for i, r in enumerate(results):
            assert r["id"] == f"seq/train{i:08d}"


class TestAutoClsTransforms:
    """Tests for image transforms applied in auto_cls."""

    def test_no_transforms_passes_image_through(self):
        """Test that images pass through untouched without transforms."""
        mock_image = {"path": "cat.png"}
        example = {"image": mock_image, "label": "cat"}

        result = auto_cls(
            example, 0, data_name="test", split="train", a_cols=["label"], transforms=[]
        )

        assert result["image"] == mock_image

    def test_transform_applied_to_pil_image(self):
        """Test that a named transform is applied to a PIL image."""
        example = {"image": _framed_image(), "label": "cat"}

        result = auto_cls(
            example,
            0,
            data_name="test",
            split="train",
            a_cols=["label"],
            transforms=["crop_black_border"],
        )

        assert isinstance(result["image"], Image.Image)
        assert result["image"].size == (4, 4)

    def test_transform_applied_to_image_dict(self):
        """Test that transforms re-encode image dicts from bytes."""
        example = {
            "image": {"path": "cat.png", "bytes": _png_bytes(_framed_image())},
            "label": "cat",
        }

        result = auto_cls(
            example,
            0,
            data_name="test",
            split="train",
            a_cols=["label"],
            transforms=["crop_black_border"],
        )

        assert result["image"]["path"] is None
        assert Image.open(BytesIO(result["image"]["bytes"])).size == (4, 4)

    def test_unknown_transform_raises(self):
        """Test that unknown transform names raise KeyError."""
        example = {"image": _framed_image(), "label": "cat"}

        with pytest.raises(KeyError, match="no_such_transform"):
            auto_cls(
                example,
                0,
                data_name="test",
                split="train",
                a_cols=["label"],
                transforms=["no_such_transform"],
            )


class TestLoadClsWithTransforms:
    """End-to-end tests for load_cls applying transforms to loaded images."""

    def test_load_cls_applies_transforms(self, tmp_path):
        """Test transforms applied to images loaded from disk."""
        ds = Dataset.from_dict(
            {"image": [_framed_image(), _framed_image()], "label": ["cat", "dog"]},
            features=DatasetFeatures(
                image=DatasetImage(decode=True), label=Value("string")
            ),
        )
        src = tmp_path / "src"
        ds.save_to_disk(src)

        args = ProcArgs(
            labels=["cat", "dog"], transforms=["crop_black_border"], show_first_n=0
        )
        pipeline = FormatterPipeline.get("auto", "cls", "train")
        result = pipeline.load(str(src), args)

        assert result[0]["image"].size == (4, 4)
        assert result[1]["image"].size == (4, 4)
        assert result[0]["label"] == 0  # ClassLabel decodes names to indices
        assert result.features["label"].names == ["cat", "dog"]

    def test_load_cls_without_transforms_keeps_images(self, tmp_path):
        """Test images stay untouched without transforms."""
        ds = Dataset.from_dict(
            {"image": [_framed_image()], "label": ["cat"]},
            features=DatasetFeatures(
                image=DatasetImage(decode=True), label=Value("string")
            ),
        )
        src = tmp_path / "src"
        ds.save_to_disk(src)

        pipeline = FormatterPipeline.get("auto", "cls", "train")
        result = pipeline.load(str(src), ProcArgs(labels=["cat"], show_first_n=0))

        assert result[0]["image"].size == (8, 8)
