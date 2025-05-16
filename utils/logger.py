"""
Shared logger utility for the agentic-retail-foundations project.
Provides a consistent logger configuration for all modules.
"""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Returns a logger with the specified name, configured with a standard format
    and INFO level by default.
    If no name is provided, returns the root logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
