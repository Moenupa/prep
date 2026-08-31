---
name: prep
description: Shared usage reference for the prep CLI, an agent-friendly converter for LLM/ML datasets backed by Hugging Face datasets. Covers CLI syntax, pipeline discovery with ppls, source loading, generic auto conversion controls, save/upload flow, and adding new formatters. Use for any prep/ppls task not specific to one target format (see sibling skills sft, verl, eval, cls, show, dev).
---

# prep CLI

`prep` converts LLM/ML datasets into common standard training and evaluation schemas backed by Hugging Face `datasets`. Format-specific details live in sibling skills: `sft`, `verl`, `eval`, `cls`, `show`.

## Installation

```sh
uv tool install prep-cli
# pip install prep-cli # fallback to pip if uv is not available

# run this to show help messages
prep --help
```

## Convert a dataset

```sh
prep TARGET_FORMAT PIPELINE_ID [SPLIT] [SOURCE[@SUBSET][:SPLIT]] [OPTIONS]
```

- `auto` is the generic conversion pipeline and the fallback for datasets without a known formatter.
  You must provide SPLIT and SOURCE (local path or Hugging Face dataset ID) to use it.
- A dataset-specific formatter pipeline, if registered, can be invoked by its ID without a source.

Examples:

```sh
prep sft auto train path/to/data.jsonl --no-save
prep eval myalias val my-org/my-dataset --q-cols question --q-cols problem --a-cols answer
```

## Available target formats

| TARGET_FORMAT |                           Usage                            | Reference       |
| ------------- | :--------------------------------------------------------: | --------------- |
| cls           |                    classification tasks                    | [cls](cls.md)   |
| eval          |                   MMMU-style evaluation                    | [eval](eval.md) |
| sft           |                   Supervised fine-tuning                   | [sft](sft.md)   |
| show          | Diagnostic load-only mode (bad samples and image warnings) | [show](show.md) |
| verl          |           VERL-compatible prompt/reward records            | [verl](verl.md) |
| dev           |     Development reference with prep-cli as dependency      | [dev](dev.md)   |

`clip` is a valid target format but has no registered pipeline or schema yet.

## General options

- `--max-samples` and `--seed`: cap the sample count and shuffle (`<0` random, `>=0` seeded).
- `--head` and `--tail`: preview the first/last N samples of the converted dataset (default: 3/0).

## Auto conversion options

Pipelines map source columns onto the target schema:

- `--q-cols` / `--a-cols`: candidate question/answer source columns; the first non-null value wins.
- `--q-template` / `--a-template`: output templates, defaulting to `{im_tags}{question}{options}` and `{answer}`. Placeholders accept `question`, `answer`, `im_tags`, `options`, and any source column name.
- `--op-cols`: candidate option columns (e.g. `options`, `choices`, `choice_a`...).
- `--extra-info`: fills the `extra_info` output field from a source column name or a literal string.

## Source loading

Loaders resolve sources in this order:

- Nonexistent local path: treat as a Hugging Face dataset ID. Use `repo` or `repo@subset` syntax.
- Existing directory: try `datasets.load_from_disk(...)`, then fall back to `load_dataset(...)` for local datasets when a split is provided.
- Existing file: infer a loader from the extension. Supported suffixes are `.parquet`, `.pq`, `.json`, `.jsonl`, `.ndjson`, `.csv`, `.tsv`, `.arrow`, `.txt`, and `.text`.

If a dataset has multiple splits, a split must be provided. `val` is automatically translated to Hugging Face `validation` when needed.

## Save and upload flow

By default, `prep` previews a few converted samples and then skips writing outputs (non-interactive mode).

- `--save`/`--no-save` forces/skips saving without prompting; `--hf`/`--no-hf` does the same for upload.
- `--save-root` changes the output root. The default is `out`.
- `--save-parq` writes a single parquet file per split instead of `save_to_disk(...)` directories.
- Preview is automatic: the converted dataset and the first `--head` / last `--tail` samples are printed, and their images are dumped under the default save path for inspection (also with `--no-save`).

Default local output paths are:

```text
out/<target_format>/<pipeline_id>/<split>
out/<target_format>/<pipeline_id>/<split>.parquet
```

The CLI also exposes Hugging Face upload options: `--hf-repo` (defaults to the pipeline ID), `--hf-subset` (defaults to `default`), `--hf-private` (defaults to true), and `--hf-nproc`. Upload targets `REPO SUBSET SPLIT` are validated and confirmed in interactive mode; with `--hf` the data is pushed as-is.

## List pipelines and outputs (`ppls`)

```sh
# ppls [OUTPUT_ROOT] [OPTIONS]
ppls
ppls out --as-json
ppls out --filter-format 's*'
ppls out --list-format
```
