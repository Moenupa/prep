from pathlib import Path

import pytest

from datasets import Dataset, DatasetDict

from prep.api.dataio import (
    OutputActions,
    PathIO,
    adaptive_load_dataset,
    load_local,
    load_remote,
    resolve_remote,
    resolve_split,
)
from prep.api.types import ProcArgs
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


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [(None, ["train"], "train"), ("test", ["train", "test"], "test"), ("val", ["validation"], "validation")],
)
def test_resolve_split_accepts_single_direct_and_validation_alias(
    requested: str | None, available: list[str], expected: str
) -> None:
    assert resolve_split(requested, available) == expected


def test_resolve_split_rejects_ambiguous_or_missing_splits() -> None:
    with pytest.raises(ValueError, match="must be specified"):
        resolve_split(None, ["train", "test"])
    with pytest.raises(ValueError, match="not found"):
        resolve_split("dev", ["train"])


def test_load_local_handles_dataset_dict_and_unloadable_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_dict({"value": [1]})
    monkeypatch.setattr("prep.api.dataio.load_from_disk", lambda _: DatasetDict(train=dataset))
    assert load_local("ignored", "train") == dataset

    monkeypatch.setattr("prep.api.dataio.load_from_disk", lambda _: (_ for _ in ()).throw(OSError("bad")))
    assert load_local("ignored") is None


def test_load_remote_parses_source_and_passes_processing_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = Dataset.from_dict({"value": [1]})
    calls: dict[str, object] = {}
    monkeypatch.setattr("prep.api.dataio.get_dataset_split_names", lambda *_: ["validation"])
    monkeypatch.setattr(
        "prep.api.dataio.load_dataset",
        lambda *args, **kwargs: calls.update(args=args, **kwargs) or expected,
    )

    assert load_remote("org/data@subset:val", args=ProcArgs(num_proc=2)) == expected
    assert calls == {"args": ("org/data", "subset"), "split": "validation", "num_proc": 2}


def test_adaptive_load_dataset_loads_files_and_applies_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("value\n1\n2\n3\n", encoding="utf-8")
    dataset = Dataset.from_dict({"value": [1, 2, 3]})
    monkeypatch.setattr("prep.api.dataio.load_file", lambda _: dataset)

    result = adaptive_load_dataset(str(source), args=ProcArgs(seed=3, max_samples=2))
    assert len(result) == 2
    assert set(result["value"]).issubset({1, 2, 3})


def test_adaptive_load_dataset_requires_remote_split_and_falls_back_to_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="Split must be specified"):
        adaptive_load_dataset("org/missing")

    local_dir = tmp_path / "downloaded"
    local_dir.mkdir()
    expected = Dataset.from_dict({"value": [1]})
    monkeypatch.setattr("prep.api.dataio.load_local", lambda *_: None)
    monkeypatch.setattr("prep.api.dataio.load_remote", lambda *_: expected)
    assert adaptive_load_dataset(str(local_dir), split="train") == expected


def test_pathio_helpers_and_tables(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sft" / "demo"
    assert PathIO.dataset_path(dataset_dir, "train", True) == dataset_dir / "train.parquet"
    assert PathIO.dataset_path(dataset_dir, "train", False) == dataset_dir / "train"
    assert not PathIO.is_overwrite(dataset_dir)
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.parquet").write_bytes(b"data")
    assert PathIO.is_overwrite(dataset_dir)
    assert PathIO.scan_splits(dataset_dir)["train"] is True
    assert PathIO.size(dataset_dir) == "4 bytes"


def test_output_actions_save_and_upload_obey_flags_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset.from_dict({"value": [1]})
    pipeline = FormatterPipeline("demo", "sft", "train", lambda *_: dataset)
    action = OutputActions(tmp_path, True, False, True, "org/repo", None, True)
    save_calls: list[object] = []
    upload_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(dataset, "save_to_disk", lambda *args, **kwargs: save_calls.append((args, kwargs)))
    monkeypatch.setattr(dataset, "push_to_hub", lambda *args, **kwargs: upload_calls.append((args, kwargs)))

    action.do_save(dataset, pipeline, nproc=2, dry_run=True)
    action.do_save(dataset, pipeline, nproc=2)
    action.do_upload(dataset, "val", nproc=2, dry_run=True)
    action.do_upload(dataset, "val", nproc=2)

    assert save_calls == [((tmp_path / "sft" / "demo" / "train",), {"num_proc": 2})]
    assert upload_calls == [(("org/repo",), {"config_name": "default", "split": "validation", "private": True, "num_proc": 2})]


def test_output_actions_skip_non_interactive_operations(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dataset = Dataset.from_dict({"value": [1]})
    pipeline = FormatterPipeline("demo", "sft", "train", lambda *_: dataset)
    action = OutputActions(tmp_path, None, False, None, "org/repo", None, False)
    action.do_save(dataset, pipeline)
    action.do_upload(dataset, "train")
    assert "Skipping (non-interactive mode)" in capsys.readouterr().out
