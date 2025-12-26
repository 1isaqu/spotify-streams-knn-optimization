"""
Preprocessing module for data loading, cleaning, and transformation.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def load_dataset(filepath: str, encoding: str = 'latin-1') -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(filepath, encoding=encoding)

def drop_irrelevant_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop specified columns from the dataframe."""
    return df.drop(columns=[c for c in columns if c in df.columns], errors='ignore')

def clean_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clean columns by removing commas and converting to numeric.
    Fills missing values with median.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            # Remove commas and convert to numeric
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            )
            # Fill NaNs with median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    return df

def fill_categorical_missing(df: pd.DataFrame, col: str = 'Artist', value: str = 'Unknown') -> pd.DataFrame:
    """Fill missing values in categorical column."""
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].fillna(value)
    return df

def apply_log_transformation(y: pd.Series) -> np.ndarray:
    """Apply log1p transformation to target variable."""
    return np.log1p(y)

def inverse_log_transformation(y_log: np.ndarray) -> np.ndarray:
    """Apply expm1 transformation to recover original scale."""
    return np.expm1(y_log)

def prepare_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target, selecting only numeric features.
    """
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols]
    y = df[target_col]
    
    return X, y
