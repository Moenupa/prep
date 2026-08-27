from pathlib import Path

import pytest

from prep.api.dataio import PathIO, resolve_remote
from prep.api.formatter import FormatterPipeline


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


class TestResolveRemote:
    """Tests for the resolve_remote() function."""

    @pytest.mark.parametrize(
        ("remote", "expected"),
        [
            ("org/dataset", ("org/dataset", None, None)),
            ("org/dataset@train", ("org/dataset", "train", None)),
            ("org/dataset:validation", ("org/dataset:validation", None, None)),
            ("org/dataset@config:test", ("org/dataset", "config", "test")),
            ("org/team/dataset", ("org/team/dataset", None, None)),
            ("org/team/dataset@subset1:val", ("org/team/dataset", "subset1", "val")),
            (
                "org/dataset@config:train-splits-001",
                ("org/dataset", "config", "train-splits-001"),
            ),
            (":", (":", None, None)),
            ("org/@dataset@extra:test", ("org/", "dataset@extra", "test")),
            ("org/dataset@sub:part1:part2", ("org/dataset", "sub", "part1:part2")),
        ],
    )
    def test_resolve_remote(
        self,
        remote: str,
        expected: tuple[str, str | None, str | None],
    ) -> None:
        """Test valid remote source parsing."""
        assert resolve_remote(remote) == expected

    @pytest.mark.parametrize(
        "remote",
        [
            pytest.param("", id="empty-string"),
            pytest.param("@", id="at-sign-alone"),
        ],
    )
    def test_invalid_source_raises(self, remote: str) -> None:
        """Test invalid source formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid source format"):
            resolve_remote(remote)
