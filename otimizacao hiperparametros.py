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
import matplotlib.pyplot as plt

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
    # Use Negative MSE for maximization logic
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return scores.mean()

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
    # delta is negative if neighbor is worse (score is lower)
    delta = neighbor_score - current_score 
    
    if neighbor_score > current_score:
        accept = True
    else:
        # No need for try/except OverflowError because delta/temp is negative and small
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

# 3. Final Evaluation
print("\n--- Final Evaluation on Test Set ---")
final_pipeline = create_knn_pipeline(best_config)

# Find the song with the highest streams (Note: df still has original values)
highest_stream_song = df.loc[df[target_col].idxmax()]
print(f"\n[INFO] Song with highest streams: {highest_stream_song['Track']} by {highest_stream_song['Artist']}")
print(f"Streams: {highest_stream_song[target_col]:,.0f}\n") 

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

# 4. Plot Evolution
plt.figure(figsize=(10, 6))
plt.plot([-h for h in history]) # Plot positive MSE
plt.title('Score Evolution (MSE - Lower is Better)')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.savefig('optimization_history.png')
print("History plot saved to 'optimization_history.png'")

# 5. Export Model
model_filename = 'best_knn_model.pkl'
joblib.dump(final_pipeline, model_filename)
print(f"Best model exported to '{model_filename}'")