import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from .config import RANDOM_STATE, TEST_SIZE, N_FEATURES_Q


def time_aware_train_test_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple time-aware split: sort by 'Time', take first (1-test_size) as train,
    the last as test.
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)
    split_index = int((1 - test_size) * n)
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Class",
) -> Dict[str, np.ndarray]:
    """
    - Split into X, y
    - Standard-scale features
    - Select K best features for quantum models
    """
    feature_cols = [c for c in train_df.columns if c not in ("Class",)]
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # For quantum models: select K best features (mutual information)
    selector = SelectKBest(mutual_info_classif, k=N_FEATURES_Q)
    X_train_q = selector.fit_transform(X_train_scaled, y_train)
    X_test_q = selector.transform(X_test_scaled)

    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]

    return {
        "X_train_full": X_train_scaled,
        "X_test_full": X_test_scaled,
        "X_train_q": X_train_q,
        "X_test_q": X_test_q,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector,
        "feature_cols": feature_cols,
        "selected_feature_names": selected_feature_names,
    }
