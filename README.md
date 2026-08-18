# prep

![PyPI Version](https://img.shields.io/pypi/v/prep-cli)
![Release](https://img.shields.io/github/v/release/Moenupa/prep)
![LICENSE](https://img.shields.io/github/license/Moenupa/prep)

`prep` is an agent-friendly CLI for converting LLM/ML datasets into common standard training and evaluation schemas backed by Hugging Face `datasets`.

It currently supports four CLI target modes:

- `sft`: two-turn OpenAI-style chat samples with images.
- `verl`: VERL-compatible prompt/reward records with images.
- `eval`: evaluation records with question, options, answer, and images.
- `show`: diagnostic load-only mode that forces dataset decoding to surface bad samples and image warnings.

## Installation

```sh
uv tool install prep-cli
# or use pip
pip install prep-cli
```

<details><summary>Install from source:</summary>

```sh
uv sync --dev
pip install -e .
```
</details>

## CLI

Convert a dataset:

```sh
prep TARGET_FORMAT PIPELINE_ID [SPLIT] [OPTIONS]
```

Examples:

```sh
# `vqa` is the generic path for datasets that already resemble common VQA formats.
# It accepts a local path or a Hugging Face dataset source and maps fields using configurable column lists and templates.
prep sft vqa train --src path/to/data.jsonl --save
prep eval vqa val --src my-org/my-dataset --q-cols question problem --a-cols answer label

# or you may use a dataset-specific formatter pipeline if one is registered:
prep verl geo3k test --save

# show a dataset, to diagnose lazy image decoding or other issues:
prep show - train --src path/to/local_dataset --no-save --no-hf
```

List registered pipelines and local output status:

```sh
ppls
ppls out --as-json
ppls out --filter-format 's*'
ppls out --list-format
```

## Source Loading

Loaders resolve sources in this order:

- Nonexistent local path: treat as a Hugging Face dataset ID. Use `repo` or `repo@subset` syntax.
- Existing directory: try `datasets.load_from_disk(...)`, then fall back to `load_dataset(...)` for local datasets when a split is provided.
- Existing file: infer a loader from the extension. Supported suffixes are `.parquet`, `.pq`, `.json`, `.jsonl`, `.ndjson`, `.csv`, `.tsv`, `.arrow`, `.txt`, and `.text`.

If a dataset has multiple splits, a split must be provided. `val` is automatically translated to Hugging Face `validation` when needed.

## Generic Conversion Controls

The generic `vqa` pipelines expose these important knobs:

- `--q-cols`: candidate question columns. Defaults to `question`, `Question`, `problem`.
- `--a-cols`: candidate answer columns. Defaults to `answer`, `Answer`, `solution`, `label`, `caption`, `correct_answer`, `reports`.
- `--op-cols`: option columns or choice fields. Defaults to `options`, `choices`, and `choice_a` through `choice_j`.
- `--q-template`: formats the prompt text. Defaults to `{im_tags}{question}{options}`.
- `--a-template`: formats the final answer text. Defaults to `{answer}`.
- `--verl-ability` and `--verl-style`: fill VERL metadata fields.
- `--max-samples`: truncate without shuffling.

Shared extraction logic supports these input patterns:

- `image`, `image_1`..`image_n`, `image_01`..`image_99`, or `images`.
- OpenAI-style `messages`.
- ShareGPT-style `conversations`.
- VERL-style `prompt` plus `reward_model`.
- Flat question/answer/option columns.

When options exist and the answer is an integer label, the mapper converts it to `A`, `B`, `C`, and so on.

## Output Schemas

`sft` produces records shaped like:

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

`verl` produces records shaped like:

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

`eval` produces records shaped like:

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

## Save And Upload Flow

By default, `prep` previews a few converted samples and then enters an interactive prompt before writing outputs.

- `--save` forces saving without prompting.
- `--no-save` skips writing altogether.
- `--save-root` changes the output root. The default is `out`.
- `--save-parq` writes a single parquet file per split instead of `save_to_disk(...)` directories.

Default local output paths are:

```text
out/<target_format>/<pipeline_id>/<split>
out/<target_format>/<pipeline_id>/<split>.parquet
```

The CLI also exposes Hugging Face upload options such as `--hf`, `--hf-repo`, `--hf-subset`, and `--hf-private`. At the moment, the upload path is wired through a dry-run call in the current implementation, so it previews the target and prompt flow but does not actually push data.

## Diagnostics

`prep show ...` is useful when a dataset fails during lazy image decoding or contains problematic files. In normal mode it runs a no-op `filter(...)` pass to force reads. If the environment variable `SLOW_PASS=1` is set, it iterates in chunks and prints the exact failing indices with warnings or errors.

The pipeline layer also validates:

- OpenAI chat structure for `sft` and `verl` samples.
- `<image>` tag counts against the number of loaded images.
- The first few converted samples through `ProcArgs.peek(...)` logging.

## Extending The Project

Add a new formatter by creating a module under `src/prep/formatter/` and registering one or more loader functions with `@formatter(...)`.

Minimal pattern:

```python
from prep.api import ProcArgs, adaptive_load_dataset, formatter


@formatter("my-dataset", "sft", "train", default_src="org/my-dataset")
def load(path: str, split: str, args: ProcArgs):
	d = adaptive_load_dataset(path, split=split, nproc=args.num_proc)
	return d.map(...)
```

Notes:

- Formatter pipeline IDs (`my-dataset`) must not contain `/`.
- `src/prep/formatter/__init__.py` auto-imports all formatter modules under `src/prep/formatter/*.py`, so registration happens on package import.
- If your dataset already follows common VQA conventions, prefer `vqa` with CLI overrides before adding a dataset-specific loader.
