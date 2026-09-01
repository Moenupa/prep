---
name: cls
description: Convert datasets to the cls format with the prep CLI — image-label classification records. Covers --labels, label anonymization via --label-resolve, and the image transform registry (--transforms or TRANSFORMS env var, e.g. crop_black_border). Use for classification data preparation with prep.
---

# prep `cls` format

Image-label classification records for classification tasks.

## Usage

```sh
prep cls auto train path/to/data --labels 'cat dog' --transforms crop_black_border
```

See the shared `prep` skill for source loading, conversion controls, and save options.

## Output schema

```python
{
    "id": str,
    "image": Image,
    "label": str,
    "extra_info": str,
}
```

`--labels` (or the `LABELS` env var) can override the `'label'` column; it should be exactly the list of class names.

## Label resolution and anonymization

`--label-resolve <to-resolve>.json` (or the `LABEL_RESOLVE` env var) points to a JSON file keyed by `'label'` (as provided to `--labels`). During `cls` conversion, case names are anonymized in favor of case IDs: the `label` column stores only integer case IDs (the index of each case name in `--labels`), and class names are dropped from the saved feature.

Two sidecar JSON files are written next to the resolve path:

- `<to-resolve>.resolved.json`: `{caseid: resolved label}` — agents can read this file directly.
- `<to-resolve>.original.json`: `{caseid: casename}` — the caseid-to-casename mapping. Agents should avoid reading this file directly, since it holds the original, un-anonymized case names.

## Image transforms

`prep` ships a registry of image-to-image transforms, currently applied during `cls` conversion by the `auto_cls` pipeline. Enable them with the `--transforms` option (or the `TRANSFORMS` env var), passing one or more transform names applied in order.

Built-in transforms:

- `convert_rgb`: converts an image to RGB mode.
- `crop_black_border`: crops away borders of black or near-black pixels (all channels at or below 10).
- `crop_black_columns`: keeps the longest horizontal run of non-black columns.
