from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from prep.api.load import (
    adaptive_load_dataset,
    load_local,
    load_remote,
    resolve_remote,
    resolve_split,
)
from prep.api.types import ProcArgs


class TestResolveRemote:
    """resolve_remote() parses ``org/data[@subset][:split]`` sources."""

    @pytest.mark.parametrize(
        ("remote", "expected"),
        [
            ("org/dataset", ("org/dataset", None, None)),
            ("org/dataset@train", ("org/dataset", "train", None)),
            ("org/dataset:validation", ("org/dataset", None, "validation")),
            ("org/dataset@config:test", ("org/dataset", "config", "test")),
            ("org/team/dataset", ("org/team/dataset", None, None)),
            ("org/team/dataset@subset1:val", ("org/team/dataset", "subset1", "val")),
            (
                "org/dataset@config:train-splits-001",
                ("org/dataset", "config", "train-splits-001"),
            ),
            ("org/@dataset@extra:test", ("org/", "dataset@extra", "test")),
            ("org/dataset@sub:part1:part2", ("org/dataset", "sub", "part1:part2")),
        ],
    )
    def test_resolve_remote(
        self,
        remote: str,
        expected: tuple[str, str | None, str | None],
    ) -> None:
        assert resolve_remote(remote) == expected

    @pytest.mark.parametrize("remote", ["", "@", ":"])
    def test_invalid_source_raises(self, remote: str) -> None:
        with pytest.raises(ValueError, match="Invalid source format"):
            resolve_remote(remote)


class TestResolveSplit:
    """resolve_split() matches requested splits against available ones."""

    @pytest.mark.parametrize(
        ("requested", "available", "expected"),
        [
            (None, ["train"], "train"),
            ("test", ["train", "test"], "test"),
            ("val", ["validation"], "validation"),
        ],
    )
    def test_resolves_requested_split(
        self, requested: str | None, available: list[str], expected: str
    ) -> None:
        assert resolve_split(requested, available) == expected

    def test_ambiguous_default_requires_explicit_split(self) -> None:
        with pytest.raises(ValueError, match="must be specified"):
            resolve_split(None, ["train", "test"])

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            resolve_split("dev", ["train"])


class TestLoadLocal:
    """load_local() wraps datasets.load_from_disk with split resolution."""

    def test_selects_split_from_dataset_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = Dataset.from_dict({"value": [1]})
        monkeypatch.setattr(
            "prep.api.load.load_from_disk",
            lambda _: DatasetDict({"train": dataset}),
        )
        assert load_local("ignored", "train") == dataset

    def test_returns_none_when_path_is_unloadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_oserror(_: object) -> Dataset:
            raise OSError("bad")

        monkeypatch.setattr("prep.api.load.load_from_disk", raise_oserror)
        assert load_local("ignored") is None


class TestLoadRemote:
    """load_remote() resolves sources and delegates to datasets.load_dataset."""

    def test_parses_source_and_passes_processing_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = Dataset.from_dict({"value": [1]})
        calls: dict[str, object] = {}
        monkeypatch.setattr(
            "prep.api.load.get_dataset_split_names", lambda *_: ["validation"]
        )
        monkeypatch.setattr(
            "prep.api.load.load_dataset",
            lambda *args, **kwargs: calls.update(args=args, **kwargs) or expected,
        )

        assert load_remote("org/data@subset:val", args=ProcArgs(num_proc=2)) == expected
        assert calls == {
            "args": ("org/data", "subset"),
            "split": "validation",
            "num_proc": 2,
        }


class TestAdaptiveLoadDataset:
    """adaptive_load_dataset() picks local/remote loaders and applies sampling."""

    def test_loads_local_file_and_applies_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "data.csv"
        source.write_text("value\n1\n2\n3\n", encoding="utf-8")
        dataset = Dataset.from_dict({"value": [1, 2, 3]})
        monkeypatch.setattr("prep.api.load.load_file", lambda _: dataset)

        result = adaptive_load_dataset(
            str(source), args=ProcArgs(seed=3, max_samples=2)
        )
        assert len(result) == 2
        assert set(result["value"]).issubset({1, 2, 3})

    def test_remote_source_requires_split(self) -> None:
        with pytest.raises(ValueError, match="Split must be specified"):
            adaptive_load_dataset("org/missing")

    def test_remote_source_with_inline_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = Dataset.from_dict({"value": [1]})
        calls: list[tuple[str, str | None]] = []

        def fake_load_remote(
            source: str, split: str | None = None, args: object = None
        ) -> Dataset:
            calls.append((source, split))
            return expected

        monkeypatch.setattr("prep.api.load.load_remote", fake_load_remote)
        assert adaptive_load_dataset("org/data@subset:train") == expected
        assert calls == [("org/data@subset:train", None)]

    def test_local_dir_falls_back_to_remote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        local_dir = tmp_path / "downloaded"
        local_dir.mkdir()
        expected = Dataset.from_dict({"value": [1]})
        monkeypatch.setattr("prep.api.load.load_local", lambda *_: None)
        monkeypatch.setattr("prep.api.load.load_remote", lambda *_: expected)
        assert adaptive_load_dataset(str(local_dir), split="train") == expected
