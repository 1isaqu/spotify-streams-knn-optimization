"""
Module for model evaluation and metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any
from src.model_training import create_knn_pipeline

def evaluate_config_cv(config: Dict[str, Any], X, y, cv_folds: int = 5) -> float:
    """
    Evaluate a configuration using Cross-Validation.
    Returns negative Mean Squared Error (higher is better for maximization logic).
    """
    pipeline = create_knn_pipeline(config)
    # n_jobs=1 to avoid conflict if this is called within multiprocessing
    scores = cross_val_score(
        pipeline, X, y, cv=cv_folds, 
        scoring='neg_mean_squared_error', n_jobs=1
    )
    return scores.mean()

def calculate_test_metrics(pipeline: Pipeline, X_test, y_test) -> Dict[str, float]:
    """Calculate metrics on test set."""
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2
    }

def create_summary_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create a summary DataFrame from optimization results."""
    summary_data = []
    for res in results:
        row = {
            'Run': res['run'],
            'Seed': res['seed'],
            'n_neighbors': res['config']['n_neighbors'],
            'weights': res['config']['weights'],
            'metric': res['config']['metric'],
            'CV_MSE': res['cv_score'], # This is actually negative MSE in the logic, but we stored positive in main loop usually
            # Note: The logic in main will convert back to positive MSE for storage
            'Test_MSE': res['test_mse'],
            'Test_RMSE': res['test_rmse'],
            'Test_MAE': res['test_mae'],
            'Test_R2': res['test_r2']
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def select_best_model(results: List[Dict], metric: str = 'test_mse') -> Dict:
    """Select the best model based on a metric (minimized)."""
    # Assuming metric should be minimized (like MSE)
    return min(results, key=lambda x: x[metric])
