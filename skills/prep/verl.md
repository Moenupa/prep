---
name: verl
description: Convert datasets to the verl format with the prep CLI — VERL-compatible prompt/reward records with images for RL training. Use when preparing VERL data, filling ability/style metadata with --verl-ability/--verl-style, or inspecting prep verl outputs.
---

# prep `verl` format

VERL-compatible prompt/reward records with images.

## Usage

```sh
# registered dataset-specific pipeline:
prep verl geo3k test --save

# generic pipeline:
prep verl auto train path/to/data.jsonl --no-save
```

VERL-specific options on the generic `auto` pipeline:

- `--verl-ability`/`--verl-style`: fill VERL metadata fields.
- `--extra-info`: fills `extra_info.misc` from a source column name or a literal string.

Answers are unwrapped from `\boxed{...}` if present, since VERL expects raw answers in `reward_model.ground_truth`.

See the shared `prep` skill for source loading, conversion controls, and save options.

## Output schema

```python
{
    "images": list[Image],
    "data_source": str,
    "prompt": [{"role": "user", "content": str}],
    "ability": str,
    "reward_model": {"style": str, "ground_truth": str},
    "extra_info": {
        "split": str,
        "index": str,
        "explanation": str,
        "misc": str,
    },
}
```

During loading, the pipeline attempts to cast to the corresponding Hugging Face `Features` object and validates a small sample window, including OpenAI chat structure for samples and `<image>` tag counts against the number of loaded images.
