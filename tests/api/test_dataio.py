from io import BytesIO
from pathlib import Path

import pytest
from datasets import Dataset, Features, Image, List
from PIL import Image as PILImage

from prep.api.dataio import (
    OutputActions,
    PathIO,
)
from prep.api.formatter import FormatterPipeline


def _png_bytes(size: int = 8) -> bytes:
    buf = BytesIO()
    PILImage.new("RGB", (size, size), "red").save(buf, format="PNG")
    return buf.getvalue()


def test_status_dicts_reports_registered_and_unregistered_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_root = tmp_path / "converted-datasets"

    registered_train_dir = save_root / "sft" / "medical-cases" / "train"
    registered_train_dir.mkdir(parents=True)
    (registered_train_dir / "sample.jsonl").write_text(
        '{"question": "What abnormality is present?", "answer": "Pleural effusion"}\n',
        encoding="utf-8",
    )

    unregistered_dataset_dir = save_root / "verl" / "scratch-pad" / "test.parquet"
    unregistered_dataset_dir.parent.mkdir(parents=True)
    unregistered_dataset_dir.write_text("placeholder parquet bytes", encoding="utf-8")

    monkeypatch.setattr(
        "prep.api.dataio.get_registered_pipelines",
        lambda: {
            ("sft", "empty-showcase"): None,
            ("sft", "medical-cases"): "team/demo-medical-cases",
        },
    )
    monkeypatch.setattr(
        FormatterPipeline,
        "unregistered_splits",
        classmethod(lambda cls, id_, target_format: {}),
    )

    reports = dict(PathIO(save_root).status_dicts())

    sft_report = reports["sft"]
    assert sft_report["ID"] == ["empty-showcase", "medical-cases"]
    assert sft_report["Default Source"] == ["", "team/demo-medical-cases"]
    assert sft_report["Local Path"] == ["", str(save_root / "sft" / "medical-cases")]
    assert sft_report["train"] == [False, True]
    assert sft_report["val"] == [False, False]
    assert sft_report["test"] == [False, False]

    unregistered_report = reports["others"]
    assert unregistered_report["ID"] == ["scratch-pad"]
    assert unregistered_report["Format"] == ["verl"]
    assert unregistered_report["Local Path"] == [
        str(save_root / "verl" / "scratch-pad")
    ]
    assert unregistered_report["train"] == [False]
    assert unregistered_report["val"] == [False]
    assert unregistered_report["test"] == [True]


def test_pathio_helpers_and_tables(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sft" / "demo"
    assert (
        PathIO.dataset_path(dataset_dir, "train", True) == dataset_dir / "train.parquet"
    )
    assert PathIO.dataset_path(dataset_dir, "train", False) == dataset_dir / "train"
    assert not PathIO.is_overwrite(dataset_dir)
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").write_bytes(b"data")
    assert PathIO.is_overwrite(dataset_dir)
    assert PathIO.scan_splits(dataset_dir)["train"] is True
    assert PathIO.size(dataset_dir) == "4 bytes"


class TestOutputActions:
    def test_save_and_upload_obey_flags_and_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = Dataset.from_dict({"value": [1]})
        pipeline = FormatterPipeline("demo", "sft", "train", lambda *_: dataset)
        action = OutputActions(
            tmp_path, True, False, None, True, "org/repo", None, True
        )
        save_calls: list[object] = []
        upload_calls: list[tuple[tuple, dict]] = []
        monkeypatch.setattr(
            dataset,
            "save_to_disk",
            lambda *args, **kwargs: save_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            dataset,
            "push_to_hub",
            lambda *args, **kwargs: upload_calls.append((args, kwargs)),
        )

        action.do_save(dataset, pipeline, nproc=2, dry_run=True)
        action.do_save(dataset, pipeline, nproc=2)
        action.do_upload(dataset, "val", nproc=2, dry_run=True)
        action.do_upload(dataset, "val", nproc=2)

        assert save_calls == [((tmp_path / "sft" / "demo" / "train",), {"num_proc": 2})]
        assert upload_calls == [
            (
                ("org/repo",),
                {
                    "config_name": "default",
                    "split": "validation",
                    "private": True,
                    "num_proc": 2,
                },
            )
        ]

    def test_output_actions_skip_non_interactive_operations(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        dataset = Dataset.from_dict({"value": [1]})
        pipeline = FormatterPipeline("demo", "sft", "train", lambda *_: dataset)
        action = OutputActions(
            tmp_path, None, False, None, None, "org/repo", None, False
        )
        action.do_save(dataset, pipeline)
        action.do_upload(dataset, "train")
        assert "Skipping (non-interactive mode)" in capsys.readouterr().out


class TestDoDump:
    """Tests for OutputActions.do_dump, which writes preview images to disk."""

    @staticmethod
    def _action(save_root: Path, preview: int) -> OutputActions:
        return OutputActions(
            save_root=save_root,
            save=False,
            save_parquet=False,
            save_preview=preview,
            hf=False,
            hf_repo="org/repo",
            hf_subset=None,
            hf_private=True,
        )

    @staticmethod
    def _blob_image_dataset(sizes: list[int]) -> Dataset:
        return Dataset.from_dict(
            {"image": [_png_bytes(size) for size in sizes]},
            features=Features(image=Image(decode=False)),
        )

    def test_dumps_undecoded_blob_images(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        dataset = self._blob_image_dataset([9, 10])
        pipeline = FormatterPipeline("demo", "cls", "train", lambda *_: dataset)

        self._action(tmp_path, preview=2).do_dump(dataset, pipeline)

        dump_dir = tmp_path / "cls" / "demo" / "train"
        assert sorted(p.name for p in dump_dir.iterdir()) == [
            "000000_00.png",
            "000001_00.png",
        ]
        for path in dump_dir.iterdir():
            with PILImage.open(path) as img:
                img.verify()
        assert capsys.readouterr().out.count("Preview image saved") == 2

    def test_dumps_decoded_pil_and_multi_image_rows(self, tmp_path: Path) -> None:
        dataset = Dataset.from_dict(
            {"images": [[_png_bytes(8)], [_png_bytes(9), _png_bytes(10)]]},
            features=Features(images=List(Image(decode=True))),
        )
        pipeline = FormatterPipeline("demo", "sft", "train", lambda *_: dataset)

        self._action(tmp_path, preview=2).do_dump(dataset, pipeline)

        dump_dir = tmp_path / "sft" / "demo" / "train"
        assert sorted(p.name for p in dump_dir.iterdir()) == [
            "000000_00.png",
            "000001_00.png",
            "000001_01.png",
        ]
        with PILImage.open(dump_dir / "000001_01.png") as img:
            assert img.size == (10, 10)

    def test_no_op_for_nonpositive_preview(self, tmp_path: Path) -> None:
        dataset = self._blob_image_dataset([8])
        pipeline = FormatterPipeline("demo", "cls", "train", lambda *_: dataset)

        self._action(tmp_path, preview=0).do_dump(dataset, pipeline)

        assert list(tmp_path.iterdir()) == []

    def test_no_op_for_empty_dataset(self, tmp_path: Path) -> None:
        dataset = Dataset.from_dict({"image": []})
        pipeline = FormatterPipeline("demo", "cls", "train", lambda *_: dataset)

        self._action(tmp_path, preview=3).do_dump(dataset, pipeline)

        assert not [p for p in tmp_path.rglob("*") if p.is_file()]

    def test_clamps_preview_count_and_honors_id_override(self, tmp_path: Path) -> None:
        dataset = self._blob_image_dataset([8])
        pipeline = FormatterPipeline("demo", "cls", "train", lambda *_: dataset)

        self._action(tmp_path, preview=10).do_dump(
            dataset, pipeline, id_override="alias"
        )

        assert [p.name for p in (tmp_path / "cls" / "alias" / "train").iterdir()] == [
            "000000_00.png"
        ]

    def test_skips_unwritable_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        dataset = self._blob_image_dataset([8])
        pipeline = FormatterPipeline("demo", "cls", "train", lambda *_: dataset)
        monkeypatch.setattr(
            "prep.api.dataio._write_preview_image", lambda entry, dest: None
        )

        self._action(tmp_path, preview=1).do_dump(dataset, pipeline)

        assert not [p for p in tmp_path.rglob("*") if p.is_file()]
        assert "Preview image saved" not in capsys.readouterr().out
