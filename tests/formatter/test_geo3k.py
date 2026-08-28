from datasets import Dataset

from prep.api import ProcArgs
from prep.formatter import geo3k


def test_geo3k_verl_loader_maps_source_fields(monkeypatch) -> None:
    source = Dataset.from_dict({"images": [[]], "problem": ["Find x"], "answer": ["2"]})
    monkeypatch.setattr(geo3k, "adaptive_load_dataset", lambda *_args, **_kwargs: source)

    result = geo3k.load("source", "test", ProcArgs(num_proc=1))
    assert result[0]["data_source"] == "geo3k"
    assert result[0]["prompt"] == [{"role": "user", "content": "Find x"}]
    assert result[0]["extra_info"]["index"] == "00000000"


def test_geo3k_sft_loader_maps_two_turn_conversation(monkeypatch) -> None:
    source = Dataset.from_dict({"images": [[]], "problem": ["Find x"], "answer": ["2"]})
    monkeypatch.setattr(geo3k, "adaptive_load_dataset", lambda *_args, **_kwargs: source)

    result = geo3k.load_sft("source", "train", ProcArgs(num_proc=1))
    assert result[0]["messages"] == [
        {"role": "user", "content": "Find x"},
        {"role": "assistant", "content": "2"},
    ]
    assert result[0]["id"] == "geo3k/00000000"
