"""
Utility functions for the fraud detection project.

This module contains helper functions for logging, configuration,
and other common tasks.
"""

from .config import Config
from .logger import setup_logger

__all__ = ["Config", "setup_logger"]
