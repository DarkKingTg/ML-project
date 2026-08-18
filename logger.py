"""
utils/logger.py

Centralized logging configuration for the BehaveGuard system. All modules
obtain their logger via `get_logger(__name__)` so log output is consistently
formatted and configurable from a single place.
"""

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def configure_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure the root logger once for the whole application. Safe to call
    multiple times — only the first call takes effect.

    Args:
        log_level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        log_file: Optional path to also write logs to a file, in addition to stdout.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name, configuring the root logger
    with sensible defaults on first use if it hasn't been configured yet.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
