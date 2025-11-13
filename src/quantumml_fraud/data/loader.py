"""
Data loading utilities for credit card fraud detection dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class DataLoader:
    """
    Loads and manages the ULB/Worldline Credit Card Fraud Detection dataset.
    
    This class provides methods to load the dataset and split it into
    training and testing sets for fraud detection modeling.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the credit card fraud detection dataset
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified path.
        
        Returns:
            DataFrame containing the credit card transaction data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2,
        random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data['Class'] if 'Class' in self.data.columns else None
        )
        
        return train_data, test_data
    
    def get_features_and_labels(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the dataset.
        
        Args:
            data: DataFrame containing the transaction data
            
        Returns:
            Tuple of (features, labels)
        """
        if 'Class' not in data.columns:
            raise ValueError("Dataset must contain 'Class' column for labels")
        
        features = data.drop('Class', axis=1).values
        labels = data['Class'].values
        
        return features, labels
