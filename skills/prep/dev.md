---
name: dev
description: Continued development reference for the prep CLI, to extend it across more dataset formatting pipelines.
---


## Extending prep as a dependency

Register one or more loader functions with `@formatter(...)`. Minimal pattern:

```py
from prep.api import ProcArgs, adaptive_load_dataset, formatter


@formatter("my-dataset", "sft", "train", default_src="org/my-dataset")
def load(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    return d.map(...)

if __name__ == "__main__":
    # redirect calls that bypass python __file__ to the CLI entrypoint
    import sys

    from typer.main import get_command

    from prep.cli import app

    # forwards arguments to the `prep` and `ppls` CLI
    # keep this in one process, otherwise the registration will be lost
    # `python script.py prep sft my-dataset train [OPTIONS]` to invoke the formatter
    get_command(app).main(args=sys.argv[1:], standalone_mode=False)
```

Notes:

- Formatter pipeline IDs (`my-dataset`) must match `[a-zA-Z0-9._-]+`.
- target formats (`sft`) must be one of the registered prep formats: `sft`, `verl`, `eval`, `clip`, or `cls`. (`show` is reserved for the diagnostic pipeline.)
