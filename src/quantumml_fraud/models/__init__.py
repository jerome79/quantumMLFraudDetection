"""
Machine learning models for fraud detection.

This module contains both classical and quantum machine learning models
for credit card fraud detection.
"""

from .classical import ClassicalMLModel
from .quantum import QuantumMLModel

__all__ = ["ClassicalMLModel", "QuantumMLModel"]
