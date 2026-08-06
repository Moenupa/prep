import logging
import os

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

NUM_PROC = int(os.getenv("NUM_PROC", "16"))


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(__name__)

    logger = logging.getLogger(name)
    return logger


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]
