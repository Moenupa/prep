import warnings
from functools import partial

from ..api import ProcArgs, adaptive_load_dataset, formatter
from ..constants import is_env_enabled

SLOW_PASS = is_env_enabled("SLOW_PASS", default="0")


def print_if_warn_or_error(start_from: int, *, d, chunksize: int) -> None:
    # chunksize to respect pre-fetching of datasets, quicker than random access
    for idx in range(start_from, min(start_from + chunksize, len(d))):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            # Ensure all warnings are captured, even if previously triggered
            warnings.simplefilter("always")
            try:
                _ = d[idx]
            except Exception as ex:
                print(f"Error at {idx}: {ex}")

            for w in recorded_warnings:
                print(f"Warning at {idx}: {w.message}")


@formatter("_", "show", "train", default_src=None)
@formatter("_", "show", "val", default_src=None)
@formatter("_", "show", "test", default_src=None)
def load(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, nproc=args.num_proc)
    if SLOW_PASS:
        # if you ever run into errors during the filtering and need to id the example
        # turn on SLOW_PASS and it will iterate and print buggy examples
        from tqdm.contrib.concurrent import process_map

        process_map(
            partial(print_if_warn_or_error, d=d, chunksize=1000),
            range(0, len(d), 1000),
            max_workers=args.num_proc,
            chunksize=1,
            desc="Checking dataset",
        )
    else:
        # dummy pass to trigger lazy loading & catch errors, e.g., pillow image decoding
        # such that you can see errors and warnings, e.g., during PILImage decoding:
        # - `UserWarning: Truncated File Read`
        # - `SyntaxError: not a TIFF file (header b'\x00\x08\x00\x04\x01\x1a\x00\x05' not valid)`
        d = d.filter(lambda e: True, num_proc=args.num_proc)

    return d
