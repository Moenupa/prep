from typing import Any


def first_value(entry: dict, keys: list[str]) -> Any | None:
    """Return the value for the first key that exists in entry."""
    for key in keys:
        if entry.get(key) is not None:
            return entry.get(key)
    return None
