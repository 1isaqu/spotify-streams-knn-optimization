"""
Configuration constants for the Spotify Streams Prediction project.
"""
import os

# Data configuration
DATA_FILENAME = "Most Streamed Spotify Songs 2024.csv"
DATA_ENCODING = 'latin-1'
TARGET_COLUMN = 'Spotify Streams'

# Columns to drop (irrelevant features)
COLUMNS_TO_DROP = [
    'SiriusXM Spins', 
    'Pandora Track Stations', 
    'Deezer Playlist Reach', 
    'Amazon Playlist Count', 
    'YouTube Playlist Reach', 
    'TIDAL Popularity'
]

# Columns to clean (remove commas and impute)
COLUMNS_TO_CLEAN = [
    'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 
    'Spotify Popularity', 
    'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 
    'TikTok Views', 'Apple Music Playlist Count', 'AirPlay Spins', 
    'Deezer Playlist Count', 'Pandora Streams', 'Soundcloud Streams', 
    'Shazam Counts'
]

# Hyperparameter space for optimization
PARAM_SPACE = {
    'n_neighbors': list(range(1, 51)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

# Optimization settings
N_RUNS = 10
SEED_START = 42
MAX_ITERATIONS = 200
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.95
CV_FOLDS = 5

# Output paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_knn_model.pkl')
COMPARISON_PLOT_PATH = os.path.join(BASE_DIR, 'optimization_history.png')
SUMMARY_CSV_PATH = os.path.join(MODELS_DIR, 'all_runs_summary.csv')
