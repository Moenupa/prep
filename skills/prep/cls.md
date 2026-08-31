---
name: cls
description: Convert datasets to the cls format with the prep CLI — image-label classification records. Covers --labels and the image transform registry (--transforms or TRANSFORMS env var, e.g. crop_black_border). Use for classification data preparation with prep.
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

`--labels` (or the `LABELS` env var) is required unless the label column is already a Hugging Face `ClassLabel` feature; it provides the class names used for casting.

## Image transforms

`prep` ships a registry of image-to-image transforms, currently applied during `cls` conversion by the `auto_cls` pipeline. Enable them with the `--transforms` option (or the `TRANSFORMS` env var), passing one or more transform names applied in order.

Built-in transforms:

- `convert_rgb`: converts an image to RGB mode.
- `crop_black_border`: crops away borders of black or near-black pixels (all channels at or below 10).
- `crop_black_columns`: keeps the longest horizontal run of non-black columns.
