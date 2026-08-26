"""Option extraction and formatting utilities."""


def get_options_from_single_entry(e: dict, option_cols: list[str]) -> list[str] | None:
    """Get options from a single entry field.

    Checks each column in option_cols and returns the first valid option list found.
    Supports both list and dict formats:
    - List: ["text A", "text B", ...]
    - Dict: {"A": "text A", "B": "text B", ...} (values sorted by key)

    Args:
        e: Example dictionary.
        option_cols: List of possible option column names.

    Returns:
        List of option strings, or None if not found.
    """
    for col in option_cols:
        options = e.get(col)
        if isinstance(options, list):
            return options
        if isinstance(options, dict):
            # return values sorted by key to ensure consistent order
            # e.g. {"A": "text A", "B": "text B", ...} -> ["text A", "text B", ...]
            # or {"b": "text B", "a": "text A", ...} -> ["text A", "text B", ...]
            return [
                v for _, v in sorted(options.items(), key=lambda kv: kv[0].casefold())
            ]

    return None


def get_options_from_multi_entry(e: dict, option_cols: list[str]) -> list[str]:
    """Get options from multiple entry fields.

    Collects option strings from multiple columns (e.g., "option_1", "option_2", ...).
    Falls back to empty list if no options found (intended for datasets without options).

    Args:
        e: Example dictionary.
        option_cols: List of possible option column names.

    Returns:
        List of option strings collected from all matching columns.
    """
    # this will fallback to an empty list if no options are found
    # which is intended for datasets without options
    options: list[str] = []
    for col in option_cols:
        if isinstance(e.get(col), str):
            options.append(e[col])
    return options


def extract_options(e: dict, option_cols: list[str]) -> list[str]:
    """Extract options from an example dictionary.

    Prefer single-entry (1&2), then falls back to multi-entry formats (3&4).
    - Case 1: e["options"] = {"A": "text A", "B": "text B", ...}
    - Case 2: e["options"] = ["text A", "text B", ...]
    - Case 3: e["option_1"] = "text A", e["option_2"] = "text B", ...
    - Case 4: e["choice_a"] = "text A", e["choice_b"] = "text B", ...

    Args:
        e: Example dictionary.
        option_cols: List of possible option column names.

    Returns:
        List of option strings.
    """
    options = get_options_from_single_entry(
        e, option_cols
    ) or get_options_from_multi_entry(e, option_cols)
    return options


def format_options(options: list[str]) -> str:
    r"""Format options as a numbered string.

    Converts a list of options into a formatted string like:
    "\nA. option 1\nB. option 2\nC. option 3"

    Args:
        options: List of option strings.

    Returns:
        Formatted string, or empty string if no options.
    """
    if not options:
        return ""
    return "\n" + (
        "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
    )
