"""
Module for creating model pipelines and training logic.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, Any

def create_knn_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Factory function to create a KNN pipeline with scaling.
    
    Args:
        config: Dictionary containing 'n_neighbors', 'weights', and 'metric'
        
    Returns:
        scikit-learn Pipeline with MinMaxScaler and KNeighborsRegressor
    """
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=config['n_neighbors'],
            weights=config['weights'],
            metric=config['metric']
        ))
    ])
