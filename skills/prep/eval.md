---
name: eval
description: Convert datasets to the eval format with the prep CLI — MMMU-style evaluation records with question, options, answer, and images. Use when preparing evaluation or benchmark datasets with prep.
---

# prep `eval` format

MMMU-style evaluation records with question, options, answer, and images.

## Usage

```sh
prep eval myalias val my-org/my-dataset --q-cols question --q-cols problem --a-cols answer
```

- Repeat `--q-cols`/`--a-cols` to pass multiple candidate columns; see the shared `prep` skill for all conversion controls and source loading details.
- When options exist and the answer is an integer label, the mapper converts it to `A`, `B`, `C`, and so on.
- Options are kept in the separate `options` field, so the `{options}` placeholder is stripped from the question template.
- Answers are unwrapped from `\boxed{...}` if present.

## Output schema

```python
{
    "id": str,
    "images": list[Image],
    "question": str,
    "options": list[str],
    "answer": str,
}
```

During loading, the pipeline attempts to cast to the corresponding Hugging Face `Features` object and validates a small sample window.
