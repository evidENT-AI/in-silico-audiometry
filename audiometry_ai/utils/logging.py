"""
Logging utilities for audiometry_ai package.

Provides consistent logging configuration across the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(name: str,
                 level: Union[str, int] = logging.INFO,
                 log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : str or int, default=logging.INFO
        Logging level
    log_file : str or Path, optional
        If provided, also log to this file

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
