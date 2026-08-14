import logging

from rich.logging import RichHandler

from . import formatter  # noqa: F401 # trigger formatter registration

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

__all__ = []
