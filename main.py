"""
Main entry point for Spotify Streams Prediction project.
Orchestrates data loading, preprocessing, optimization, and evaluation.
"""
import os
import argparse
from sklearn.model_selection import train_test_split
import config
from src import preprocessing, hyperopt, evaluation, utils_io

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Spotify Streams KNN Optimization')
    parser.add_argument('--n-runs', type=int, default=config.N_RUNS, help='Number of optimization runs')
    parser.add_argument('--seed', type=int, default=config.SEED_START, help='Starting random seed')
    parser.add_argument('--max-iters', type=int, default=config.MAX_ITERATIONS, help='Max iterations per run')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SPOTIFY STREAMS PREDICTION - HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}\n")
    
    # 1. Load and Preprocess
    print("[1/5] Loading and preprocessing data...")
    print(f"      File: {config.DATA_FILENAME}")
    
    data_path = os.path.join(config.BASE_DIR, config.DATA_FILENAME)
    if not os.path.exists(data_path):
        print(f"ERROR: File not found at {data_path}")
        return

    df = preprocessing.load_dataset(data_path, config.DATA_ENCODING)
    print(f"      Original shape: {df.shape}")
    
    # Cleaning pipeline
    df = preprocessing.drop_irrelevant_columns(df, config.COLUMNS_TO_DROP)
    df = preprocessing.clean_numeric_columns(df, config.COLUMNS_TO_CLEAN)
    df = preprocessing.fill_categorical_missing(df)
    
    X, y = preprocessing.prepare_features_target(df, config.TARGET_COLUMN)
    y_log = preprocessing.apply_log_transformation(y)
    
    print(f"      Processed shape: {X.shape}")
    print("      Target variable log-transformed.")

    # 2. Train/Test Split
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    print(f"      Train set: {X_train.shape[0]} samples")
    print(f"      Test set:  {X_test.shape[0]} samples")

    # 3. Optimization
    print(f"\n[3/5] Running Multi-Run Optimization...")
    print(f"      Runs: {args.n_runs}, Parallel execution")
    print(f"      Param Space: {config.PARAM_SPACE}")
    
    results = hyperopt.run_multi_optimization_parallel(
        n_runs=args.n_runs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        param_space=config.PARAM_SPACE,
        max_iters=args.max_iters,
        initial_temp=config.INITIAL_TEMPERATURE,
        cooling_rate=config.COOLING_RATE,
        start_seed=args.seed
    )

    # 4. Save individual results
    print("\n[4/5] Saving individual run artifacts...")
    utils_io.ensure_directory_exists(config.MODELS_DIR)
    
    for res in results:
        # Save individual model
        model_path = os.path.join(config.MODELS_DIR, f"knn_model_run{res['run']}_seed{res['seed']}.pkl")
        utils_io.save_model(res['pipeline'], model_path)
        
        # Save individual plot
        plot_path = os.path.join(config.MODELS_DIR, f"optimization_history_run{res['run']}.png")
        utils_io.save_run_plot(res['history'], res['run'], plot_path)
    
    utils_io.save_results_csv(results, config.SUMMARY_CSV_PATH)
    utils_io.save_comparison_plot(results, config.COMPARISON_PLOT_PATH)

    # 5. Final Selection
    print("\n[5/5] Final Evaluation...")
    best_result = evaluation.select_best_model(results, metric='test_mse')
    
    # Save best model to root
    utils_io.save_model(best_result['pipeline'], config.BEST_MODEL_PATH)

    # Print Report
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Run:       Run {best_result['run']} (Seed {best_result['seed']})")
    print(f"Best Config:    {best_result['config']}")
    print(f"{'-'*30}")
    print(f"Test MSE:       {best_result['test_mse']:.4f}")
    print(f"Test RMSE:      {best_result['test_rmse']:.4f}")
    print(f"Test R²:        {best_result['test_r2']:.4f}")
    print(f"{'-'*30}")
    print(f"Saved best model to: {config.BEST_MODEL_PATH}")
    print(f"Saved comparison plot to: {config.COMPARISON_PLOT_PATH}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
