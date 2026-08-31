# prep

![PyPI Version](https://img.shields.io/pypi/v/prep-cli)
![Release](https://img.shields.io/github/v/release/Moenupa/prep)
![LICENSE](https://img.shields.io/github/license/Moenupa/prep)

`prep` is an agent-friendly CLI for converting LLM/ML datasets into common
standard training and evaluation schemas backed by Hugging Face `datasets`.

## Installation

```sh
uv tool install prep-cli
# pip install prep-cli # or pip
prep --help
```

<details><summary>Alternatively, you can install from source:</summary>

```sh
# as tool
uv tool install 'prep-cli @ https://github.com/Moenupa/prep.git'
prep --help

# as folder
git clone https://github.com/Moenupa/prep.git
uv sync --dev
uv run prep --help
```
</details>

## Usage

```sh
# export UI=1 # if you want interactive access
prep TARGET_FORMAT PIPELINE_ID [SPLIT] [SOURCE[@SUBSET][:SPLIT]] [OPTIONS]
```

```sh
# `auto`: generic conversion pipeline, requires SPLIT and SOURCE:
prep sft auto train path/to/data.jsonl --no-save

# a registered dataset-specific formatter pipeline:
prep verl geo3k test --save

# list registered pipelines and local output status:
ppls
```

## Documentation

The full usage reference lives in [skills/prep](skills/prep/SKILL.md),
maintained for both humans and coding agents:

- [SKILL.md](skills/prep/SKILL.md) — shared usage: CLI syntax, source loading, auto conversion controls, save/upload flow, and `ppls`.
- [sft.md](skills/prep/sft.md) / [verl.md](skills/prep/verl.md) / [eval.md](skills/prep/eval.md) / [cls.md](skills/prep/cls.md) — per-format options and output schemas.
- [show.md](skills/prep/show.md) — dataset diagnostics.
- [dev.md](skills/prep/dev.md) — extending `prep` with new formatter pipelines.
