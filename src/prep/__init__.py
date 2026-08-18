import logging

from rich.logging import RichHandler

from . import formatter  # noqa: F401 # trigger formatter registration

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, show_path=False, rich_tracebacks=True)],
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.ERROR)

__all__ = []
