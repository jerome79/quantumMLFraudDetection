"""Data loading utilities for fraud detection dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import config


def load_creditcard_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset.
    
    Args:
        filepath: Path to creditcard.csv file. If None, uses default from config.
        
    Returns:
        DataFrame containing the credit card transaction data.
    """
    if filepath is None:
        filepath = config.DATASET_PATH
    
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


def get_train_test_split(df: pd.DataFrame, 
                         test_size: float = None,
                         random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['Class']
    )
    
    print(f"Training set size: {train_df.shape}")
    print(f"Test set size: {test_df.shape}")
    
    return train_df, test_df


def sample_balanced_dataset(df: pd.DataFrame, 
                           sample_size: int = None,
                           random_state: int = None) -> pd.DataFrame:
    """
    Create a balanced sample from the dataset for quantum models.
    
    This is useful for quantum models which have computational constraints.
    
    Args:
        df: Input DataFrame
        sample_size: Total number of samples (split equally between classes)
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced DataFrame with sample_size/2 fraud and sample_size/2 non-fraud cases
    """
    if sample_size is None:
        sample_size = config.SAMPLE_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    fraud_df = df[df['Class'] == 1]
    non_fraud_df = df[df['Class'] == 0]
    
    samples_per_class = sample_size // 2
    
    # Sample from each class
    fraud_sample = fraud_df.sample(
        n=min(samples_per_class, len(fraud_df)),
        random_state=random_state
    )
    non_fraud_sample = non_fraud_df.sample(
        n=min(samples_per_class, len(non_fraud_df)),
        random_state=random_state
    )
    
    # Combine and shuffle
    balanced_df = pd.concat([fraud_sample, non_fraud_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Created balanced sample with {len(balanced_df)} samples")
    print(f"Fraud ratio: {balanced_df['Class'].mean()*100:.2f}%")
    
    return balanced_df
