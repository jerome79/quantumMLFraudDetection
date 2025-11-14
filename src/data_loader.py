"""Data loading utilities for fraud detection dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from . import config


def load_creditcard_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset.
    
    Args:
        filepath: Path to creditcard.csv file. If None, uses default from config.
        
    Returns:
        DataFrame containing the credit card transaction data.
    """
    if filepath is None:
        filepath = config.DATA_PATH
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download the ULB/Worldline Credit Card Fraud Detection dataset "
            f"and place it in {config.RAW_DATA_DIR}/"
        )
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    return df
