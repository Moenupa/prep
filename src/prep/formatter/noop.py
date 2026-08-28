import warnings

from ..api import ProcArgs, adaptive_load_dataset, formatter, get_logger

logger = get_logger(__name__)


def print_if_warn_or_error(start_from: int, *, d, chunksize: int) -> None:
    # chunksize to respect pre-fetching of datasets, quicker than random access
    for idx in range(start_from, min(start_from + chunksize, len(d))):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            # Ensure all warnings are captured, even if previously triggered
            warnings.simplefilter("always")
            try:
                _ = d[idx]
            except Exception as ex:
                logger.error("Error at %d: %s", idx, ex)

            for w in recorded_warnings:
                logger.warning("Warning at %d: %s", idx, w.message)


@formatter("_", "show", "train", default_src=None)
@formatter("_", "show", "val", default_src=None)
@formatter("_", "show", "test", default_src=None)
def load(path: str, split: str, args: ProcArgs):
    d = adaptive_load_dataset(path, split=split, args=args)
    from functools import partial

    from tqdm.contrib.concurrent import process_map

    chunksize = max(1000, len(d) // (args.num_proc * 4))
    process_map(
        partial(print_if_warn_or_error, d=d, chunksize=chunksize),
        range(0, len(d), chunksize),
        max_workers=args.num_proc,
        chunksize=1,
        desc="Checking dataset",
    )
    # dummy pass to trigger lazy loading & catch errors, e.g., pillow image decoding
    # such that you can see errors and warnings, e.g., during PILImage decoding:
    # - `UserWarning: Truncated File Read`
    # - `SyntaxError: not a TIFF file (header b'\x00\x08\x00\x04\x01\x1a\x00\x05' not valid)`
    # d = d.filter(lambda e: True, num_proc=args.num_proc)

    return d
