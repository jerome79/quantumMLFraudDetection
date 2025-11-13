"""Utility functions for the fraud detection project."""

import os
import numpy as np
import random
import pickle
from pathlib import Path
from typing import Any


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def create_directory(path: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_object(obj: Any, filepath: str):
    """
    Save a Python object using pickle.
    
    Args:
        obj: Object to save
        filepath: Path to save the object
    """
    # Create directory if needed
    directory = os.path.dirname(filepath)
    if directory:
        create_directory(directory)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_object(filepath: str) -> Any:
    """
    Load a Python object using pickle.
    
    Args:
        filepath: Path to the saved object
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj


def print_dataset_info(df):
    """
    Print information about the dataset.
    
    Args:
        df: Pandas DataFrame
    """
    print("\nDataset Information:")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    print("-" * 40)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def check_dataset_exists(filepath: Path) -> bool:
    """
    Check if dataset file exists.
    
    Args:
        filepath: Path to dataset file
        
    Returns:
        True if file exists, False otherwise
    """
    exists = filepath.exists()
    if exists:
        print(f"✓ Dataset found at {filepath}")
    else:
        print(f"✗ Dataset not found at {filepath}")
        print(f"  Please download the ULB Credit Card Fraud Detection dataset")
        print(f"  and place it at: {filepath}")
    return exists
