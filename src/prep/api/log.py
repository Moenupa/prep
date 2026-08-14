import logging


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(__name__)

    logger = logging.getLogger(name)
    return logger
