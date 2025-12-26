"""
Hyperparameter optimization module using Random Hill Climbing and Simulated Annealing.
"""
import random
import math
import numpy as np
from typing import Dict, Any, Tuple, List
from multiprocessing import Pool, cpu_count

from src.evaluation import evaluate_config_cv, calculate_test_metrics
from src.model_training import create_knn_pipeline

class ParamSpace:
    """Manages hyperparameter search space."""
    
    def __init__(self, space_dict: Dict[str, List]):
        self.space = space_dict
        
    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        return {k: random.choice(v) for k, v in self.space.items()}
    
    def get_neighbor(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a neighbor configuration by changing one parameter."""
        new_config = config.copy()
        param_to_change = random.choice(list(self.space.keys()))
        new_config[param_to_change] = random.choice(self.space[param_to_change])
        return new_config

def run_simulated_annealing(
    X_train, y_train, 
    param_space: Dict[str, List],
    max_iters: int = 200,
    initial_temp: float = 1000,
    cooling_rate: float = 0.95
) -> Tuple[Dict, float, List[float]]:
    """Run Simulated Annealing optimization."""
    
    space = ParamSpace(param_space)
    
    # 1. Initialization: Find best random start
    best_config = None
    best_score = -float('inf')  # We maximize negative MSE (minimize MSE)
    
    # Try 20 initial configurations to find a good start
    for _ in range(20):
        config = space.sample()
        score = evaluate_config_cv(config, X_train, y_train)
        if score > best_score:
            best_score = score
            best_config = config
    
    current_config = best_config
    current_score = best_score
    history = [-best_score] # Store positive MSE for history
    
    temperature = initial_temp
    
    # 2. Optimization Loop
    for _ in range(max_iters):
        neighbor = space.get_neighbor(current_config)
        neighbor_score = evaluate_config_cv(neighbor, X_train, y_train)
        
        # Calculate acceptance probability
        delta = neighbor_score - current_score
        
        if delta > 0:
            # Better solution: always accept
            current_config = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_config = current_config
        else:
            # Worse solution: probabilistic acceptance
            prob = math.exp(delta / temperature)
            if random.random() < prob:
                current_config = neighbor
                current_score = neighbor_score
        
        history.append(-best_score)
        temperature *= cooling_rate
        
    return best_config, best_score, history

def run_single_optimization_task(args):
    """Worker function for multiprocessing."""
    (run_id, seed, X_train, y_train, X_test, y_test, 
     param_space, max_iters, initial_temp, cooling_rate) = args
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Run optimization
    best_config, best_cv_score, history = run_simulated_annealing(
        X_train, y_train, param_space, max_iters, initial_temp, cooling_rate
    )
    
    # Final evaluation on test set
    final_pipeline = create_knn_pipeline(best_config)
    final_pipeline.fit(X_train, y_train)
    test_metrics = calculate_test_metrics(final_pipeline, X_test, y_test)
    
    return {
        'run': run_id,
        'seed': seed,
        'config': best_config,
        'cv_score': -best_cv_score, # Convert back to positive MSE
        'history': history,
        'test_mse': test_metrics['test_mse'],
        'test_rmse': test_metrics['test_rmse'],
        'test_mae': test_metrics['test_mae'],
        'test_r2': test_metrics['test_r2'],
        'pipeline': final_pipeline
    }

def run_multi_optimization_parallel(
    n_runs, X_train, y_train, X_test, y_test, 
    param_space, max_iters, initial_temp, cooling_rate, start_seed=42
):
    """Orchestrate multiple parallel optimization runs."""
    
    cpu_cores = min(n_runs, cpu_count())
    print(f"Starting {n_runs} runs on {cpu_cores} cores...")
    
    tasks = []
    for i in range(1, n_runs + 1):
        seed = start_seed + i - 1
        tasks.append((
            i, seed, X_train, y_train, X_test, y_test,
            param_space, max_iters, initial_temp, cooling_rate
        ))
    
    with Pool(processes=cpu_cores) as pool:
        results = pool.map(run_single_optimization_task, tasks)
        
    return results
