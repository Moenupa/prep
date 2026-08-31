---
name: show
description: Diagnose datasets with prep show — a load-only mode that forces dataset decoding to surface bad samples, lazy image decoding failures, and warnings. Use when a dataset fails during prep conversion or image decoding.
---

# prep `show` — dataset diagnostics

`prep show ...` is a diagnostic load-only mode that forces dataset decoding to surface bad samples and image warnings. It iterates through the dataset and prints the exact failing indices with warnings or errors.

## Usage

```sh
prep show - train path/to/local_dataset --no-save --no-hf
```

`-` is the pipeline ID placeholder since no conversion happens. See the shared `prep` skill for source loading and save options.

## What the pipeline layer also validates

- OpenAI chat structure for `sft` and `verl` samples.
- `<image>` tag counts against the number of loaded images.
- The first few converted samples through `ProcArgs.peek(...)` logging.
