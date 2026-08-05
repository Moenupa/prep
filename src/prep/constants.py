import logging
import os

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    datefmt="[%m/%d %H:%M]",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_LOGGER = logging.getLogger(__name__)
NUM_PROC = int(os.getenv("NUM_PROC", "16"))
OUT_DIR = os.getenv("PREP_DIR", os.path.expanduser("~/.cache/prep"))


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return DEFAULT_LOGGER

    logger = logging.getLogger(name)
    return logger


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    """Check whether an environment variable is set to a truthy value.

    Truthy values are ``"true"``, ``"yes"``, ``"on"``, ``"t"``, ``"y"``, ``"1"``
    (case-insensitive).

    Args:
        env_var (str): Name of the environment variable to inspect.
        default (str, optional): Value to use when the variable is unset.
            (default: ``"0"``)

    Returns:
        bool: ``True`` if the resolved value is truthy, ``False`` otherwise.
    """
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]
