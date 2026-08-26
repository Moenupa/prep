"""Unit tests for auto_cls formatter without requiring Hugging Face datasets."""

import pytest

from prep.formatter.auto_cls import auto_cls


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

        result = auto_cls(example, 0, data_name="test", split="train", a_cols=["label"])

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
            auto_cls(example, 0, data_name="test", split="train", a_cols=["label"])

    def test_auto_cls_empty_extra_info(self):
        """Test classification formatting handles missing extra_info."""
        mock_image = {"path": "dog.png"}
        example = {
            "image": mock_image,
            "label": "dog",
        }

        result = auto_cls(example, 5, data_name="pets", split="test", a_cols=["label"])

        assert result["extra_info"] == ""

    def test_auto_cls_id_format_with_split(self):
        """Test that ID includes split information."""
        mock_image = {"path": "bird.png"}
        example = {
            "image": mock_image,
            "label": "bird",
        }

        result_train = auto_cls(
            example, 0, data_name="birds", split="train", a_cols=["label"]
        )
        result_val = auto_cls(
            example, 1, data_name="birds", split="val", a_cols=["label"]
        )
        result_test = auto_cls(
            example, 2, data_name="birds", split="test", a_cols=["label"]
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
            example, 10, data_name="cifar", split="test", a_cols=["label"]
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
            example, 0, data_name="cifar10", split="test", a_cols=["category"]
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
        result = auto_cls(example, 0, data_name="test", split="train", a_cols=["label"])

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
                example, i, data_name="seq", split="train", a_cols=["label"]
            )
            results.append(result)

        for i, r in enumerate(results):
            assert r["id"] == f"seq/train{i:08d}"
