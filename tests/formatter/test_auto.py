import pytest

from prep.formatter.auto import load_eval, load_sft, load_verl


@pytest.mark.slow
@pytest.mark.parametrize(
    "path, split, expected",
    [
        (
            "cais/mmlu@abstract_algebra",
            "test",
            {
                "id": "mmlu@abstract_algebra/test00000000",
                "images": [],
                "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
                "options": ["0", "4", "2", "6"],
                "answer": 1,
            },
        ),
    ],
)
def test_load_eval(path, split, procargs, expected):
    assert load_eval(path, split, procargs)[0] == expected


AIME24_Q1 = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."


@pytest.mark.slow
@pytest.mark.parametrize(
    "path, split, expected",
    [
        (
            "math-ai/aime24",
            "test",
            {
                "images": [],
                "data_source": "aime24",
                "prompt": [
                    {
                        "role": "user",
                        "content": AIME24_Q1,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": "204"},
                "extra_info": {
                    "split": "test",
                    "index": "00000000",
                    "explanation": "",
                    "misc": "",
                },
            },
        ),
    ],
)
def test_load_verl(path, split, procargs, expected):
    assert load_verl(path, split, procargs)[0] == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "path, split, expected",
    [
        (
            "math-ai/aime24",
            "test",
            {
                "images": [],
                "messages": [
                    {"role": "user", "content": AIME24_Q1},
                    {"role": "assistant", "content": "\\boxed{204}"},
                ],
                "id": "aime24/00000000",
                "extra_info": "",
            },
        ),
    ],
)
def test_load_sft(path, split, procargs, expected):
    assert load_sft(path, split, procargs)[0] == expected
