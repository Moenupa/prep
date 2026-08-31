---
name: sft
description: Convert datasets to the sft format with the prep CLI — two-turn OpenAI-style chat samples with images for supervised fine-tuning. Use when converting or preparing SFT training data with prep.
---

# prep `sft` format

Two-turn OpenAI-style chat samples with images.

## Usage

```sh
prep sft auto train path/to/data.jsonl --no-save
```

`auto` is the generic pipeline; see the shared `prep` skill for source loading, conversion controls (`--q-cols`, `--a-cols`, templates, `--max-samples`, `--extra-info`), and save options.

## Output schema

```python
{
    "images": list[Image],
    "messages": [
        {"role": "user", "content": str},
        {"role": "assistant", "content": str},
    ],
    "id": str,
    "extra_info": str,
}
```

During loading, the pipeline attempts to cast to the corresponding Hugging Face `Features` object and validates a small sample window, including OpenAI chat structure for samples and `<image>` tag counts against the number of loaded images.
