"""Label extraction utilities."""

from .helpers import first_value


def extract_label(e: dict, label_cols: list[str]) -> str:
    """Extract label from an example dictionary.

    Gets the first non-null value from the specified label columns.

    Args:
        e: Example dictionary.
        label_cols: List of possible label column names.

    Returns:
        The label string.

    Raises:
        ValueError: If label is None or not found.
    """
    label = first_value(e, label_cols)
    if label is not None:
        return label

    raise ValueError(f"Expected label in one of {label_cols}, got None in example: {e}")
