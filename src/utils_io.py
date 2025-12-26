"""
I/O utilities for file handling, plotting, and model persistence.
"""
import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict

# Use non-interactive backend
matplotlib.use('Agg')

def ensure_directory_exists(path: str):
    """Ensure that the directory for a given path exists."""
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_model(pipeline, filepath: str):
    """Save trained model to disk."""
    ensure_directory_exists(filepath)
    joblib.dump(pipeline, filepath)
    print(f"Model saved to '{filepath}'")

def load_model(filepath: str):
    """Load trained model from disk."""
    return joblib.load(filepath)

def save_results_csv(results: List[Dict], filepath: str):
    """Save results summary to CSV."""
    from src.evaluation import create_summary_dataframe
    ensure_directory_exists(filepath)
    
    df = create_summary_dataframe(results)
    df.to_csv(filepath, index=False)
    print(f"Summary saved to '{filepath}'")

def save_run_plot(history: List[float], run_id: int, filepath: str):
    """Save optimization history plot for a single run."""
    ensure_directory_exists(filepath)
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(f'Optimization History - Run {run_id}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()

def save_comparison_plot(results: List[Dict], filepath: str):
    """Save comparison plot for multiple runs."""
    ensure_directory_exists(filepath)
    
    runs = [r['run'] for r in results]
    mses = [r['test_mse'] for r in results]
    r2s = [r['test_r2'] for r in results]
    
    plt.figure(figsize=(15, 6))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.bar(runs, mses, color='steelblue')
    plt.title('Test MSE Across All 10 Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Test MSE')
    plt.grid(True, alpha=0.3)
    
    # Plot R2
    plt.subplot(1, 2, 2)
    plt.bar(runs, r2s, color='forestgreen')
    plt.title('Test R² Across All 10 Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Test R²')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Comparison plot saved to '{filepath}'")
