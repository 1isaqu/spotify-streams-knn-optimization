import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import random
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count

# Load the dataset with latin-1 encoding
df = pd.read_csv(r"Most Streamed Spotify Songs 2024.csv", encoding='latin-1')

# 1. Drop specified columns
cols_to_drop = [
    'SiriusXM Spins', 
    'Pandora Track Stations', 
    'Deezer Playlist Reach', 
    'Amazon Playlist Count', 
    'YouTube Playlist Reach', 
    'TIDAL Popularity'
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# 2. Columns to clean (remove commas) and impute with median
cols_to_clean = [
    'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 
    'Spotify Popularity', 
    'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 
    'TikTok Views', 'Apple Music Playlist Count', 'AirPlay Spins', 
    'Deezer Playlist Count', 'Pandora Streams', 'Soundcloud Streams', 
    'Shazam Counts'
]

for col in cols_to_clean:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaNs
df['Artist'] = df['Artist'].fillna('Unknown')


# --- KNN Hyperparameter Optimization (Random Hill Climbing / Simulated Annealing) ---

# Prepare data
target_col = 'Spotify Streams'
print(f"Target Column: {target_col}")

# Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

X = df[numeric_cols]
# Apply Log Transformation to the Target
y = np.log1p(df[target_col])
print("Target variable 'Spotify Streams' has been Log-transformed (np.log1p).")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Space
param_space = {
    'n_neighbors': list(range(1, 51)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

def create_knn_pipeline(config):
    """Factory function to create the KNN pipeline."""
    return Pipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=config['n_neighbors'],
            weights=config['weights'],
            metric=config['metric']
        ))
    ])

def sample_param(param):
    return random.choice(param_space[param])

def get_random_config():
    return {p: sample_param(p) for p in param_space}

def get_neighbor(config):
    new_config = config.copy()
    param_to_change = random.choice(list(param_space.keys()))
    new_config[param_to_change] = sample_param(param_to_change)
    return new_config

def evaluate_model(config, X, y):
    pipeline = create_knn_pipeline(config)
    # Use n_jobs=1 to avoid conflict with multiprocessing at run level
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
    return scores.mean()

def run_single_optimization(args):
    """Run a single optimization with a specific seed."""
    run_num, seed, X_train, X_test, y_train, y_test = args
    
    # Set seed for reproducibility of this specific run
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"RUN {run_num}/10 - SEED: {seed}")
    print(f"{'='*80}")
    
    print("\n--- Starting Random Hill Climbing with Simulated Annealing ---")
    
    # 1. Initial Population
    print("Generating 20 initial random configurations...")
    best_config = None
    best_score = -float('inf')
    
    for _ in range(20):
        config = get_random_config()
        score = evaluate_model(config, X_train, y_train)
        if score > best_score:
            best_score = score
            best_config = config
    
    current_config = best_config
    current_score = best_score
    history = [current_score]
    
    print(f"Initial Best Config: {current_config}")
    print(f"Initial Best Score (MSE): {-current_score:.4f}")
    
    # 2. Hill Climbing Loop
    max_iterations = 200
    temperature = 1000  # Initial temperature for Simulated Annealing
    cooling_rate = 0.95
    
    print(f"Running {max_iterations} iterations...")
    for i in range(max_iterations):
        # Generate neighbor
        neighbor_config = get_neighbor(current_config)
        neighbor_score = evaluate_model(neighbor_config, X_train, y_train)
        
        # Acceptance Probability (Simulated Annealing)
        delta = neighbor_score - current_score 
        
        if neighbor_score > current_score:
            accept = True
        else:
            probability = math.exp(delta / temperature)
            accept = random.random() < probability

        if accept:
            current_config = neighbor_config
            current_score = neighbor_score
            # Update overall best
            if current_score > best_score:
                best_score = current_score
                best_config = current_config
                print(f"Iter {i+1}: New Best found! {best_config} Score (MSE): {-best_score:.4e}")
        
        history.append(current_score)
        temperature *= cooling_rate
    
    print("\n--- Optimization Finished ---")
    print(f"Best Config: {best_config}")
    print(f"Best CV Score (MSE): {-best_score:.4f}")
    
    # 3. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    final_pipeline = create_knn_pipeline(best_config)
    final_pipeline.fit(X_train, y_train)
    y_pred = final_pipeline.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"1. MSE:  {mse:.4e}")
    print(f"2. RMSE: {rmse:.4e}")
    print(f"3. MAE:  {mae:.4e}")
    print(f"4. R2:   {r2:.4f}")
    
    # 4. Save model and history plot
    model_filename = f'models/knn_model_run{run_num}_seed{seed}.pkl'
    joblib.dump(final_pipeline, model_filename)
    print(f"Model saved to '{model_filename}'")
    
    # Save history plot
    plt.figure(figsize=(10, 6))
    plt.plot([-h for h in history])
    plt.title(f'Run {run_num} - Score Evolution (MSE - Lower is Better) - Seed: {seed}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plot_filename = f'models/optimization_history_run{run_num}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"History plot saved to '{plot_filename}'")
    
    # Return results
    return {
        'run': run_num,
        'seed': seed,
        'config': best_config,
        'cv_score': -best_score,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'model_path': model_filename,
        'pipeline': final_pipeline
    }

# Create directory for models
os.makedirs('models', exist_ok=True)

print("\n" + "="*80)
print("RUNNING 10 INDEPENDENT OPTIMIZATION RUNS IN PARALLEL")
print(f"Using {cpu_count()} CPU cores")
print("="*80)

# Prepare arguments for parallel processing
run_args = [(run_num, 42 + run_num, X_train, X_test, y_train, y_test) 
            for run_num in range(1, 11)]

# Run optimizations in parallel
if __name__ == '__main__':
    with Pool(processes=min(10, cpu_count())) as pool:
        all_runs_results = pool.map(run_single_optimization, run_args)

    # ============================================================================
    # SELECT BEST MODEL ACROSS ALL RUNS
    # ============================================================================

    print("\n" + "="*80)
    print("SUMMARY OF ALL 10 RUNS")
    print("="*80)

    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'Run': r['run'],
        'Seed': r['seed'],
        'n_neighbors': r['config']['n_neighbors'],
        'weights': r['config']['weights'],
        'metric': r['config']['metric'],
        'CV_MSE': r['cv_score'],
        'Test_MSE': r['test_mse'],
        'Test_RMSE': r['test_rmse'],
        'Test_MAE': r['test_mae'],
        'Test_R2': r['test_r2']
    } for r in all_runs_results])

    print("\n" + summary_df.to_string(index=False))

    # Save summary to CSV
    summary_df.to_csv('models/all_runs_summary.csv', index=False)
    print("\nSummary saved to 'models/all_runs_summary.csv'")

    # Find best model based on Test MSE (lower is better)
    best_run = min(all_runs_results, key=lambda x: x['test_mse'])

    print("\n" + "="*80)
    print("BEST MODEL SELECTED")
    print("="*80)
    print(f"Run Number: {best_run['run']}")
    print(f"Seed: {best_run['seed']}")
    print(f"Configuration: {best_run['config']}")
    print(f"CV MSE: {best_run['cv_score']:.4e}")
    print(f"Test MSE: {best_run['test_mse']:.4e}")
    print(f"Test RMSE: {best_run['test_rmse']:.4e}")
    print(f"Test MAE: {best_run['test_mae']:.4e}")
    print(f"Test R2: {best_run['test_r2']:.4f}")

    # Save the best model to the main directory
    best_model_filename = 'best_knn_model.pkl'
    joblib.dump(best_run['pipeline'], best_model_filename)
    print(f"\nBest model exported to '{best_model_filename}'")

    # Create a comparison plot of all runs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 11), [r['test_mse'] for r in all_runs_results], color='steelblue')
    plt.xlabel('Run Number')
    plt.ylabel('Test MSE')
    plt.title('Test MSE Across All 10 Runs')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 11), [r['test_r2'] for r in all_runs_results], color='forestgreen')
    plt.xlabel('Run Number')
    plt.ylabel('Test R²')
    plt.title('Test R² Across All 10 Runs')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_history.png')
    print("Comparison plot saved to 'optimization_history.png'")

    # Find the song with the highest streams
    highest_stream_song = df.loc[df[target_col].idxmax()]
    print(f"\n[INFO] Song with highest streams: {highest_stream_song['Track']} by {highest_stream_song['Artist']}")
    print(f"Streams: {highest_stream_song[target_col]:,.0f}")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)