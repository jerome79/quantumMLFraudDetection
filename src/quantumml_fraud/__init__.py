"""
Quantum Machine Learning for Credit Card Fraud Detection

This package provides tools for fraud detection using both classical
and quantum machine learning approaches on credit card transaction data.
"""

__version__ = "0.1.0"
__author__ = "jerome79"

from . import data
from . import models
from . import evaluation
from . import utils

__all__ = ["data", "models", "evaluation", "utils"]
