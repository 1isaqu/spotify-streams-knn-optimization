# Spotify Streams Prediction - KNN Hyperparameter Optimization

## 📊 Project Overview

This project implements a **K-Nearest Neighbors (KNN) Regressor** optimized using **Random Hill Climbing with Simulated Annealing** to predict Spotify streaming numbers. The model is trained on the "Most Streamed Spotify Songs 2024" dataset from Kaggle and achieves robust performance through advanced hyperparameter tuning techniques.

## 🎯 Objectives

- Predict Spotify streaming counts based on various music platform metrics
- Optimize KNN hyperparameters using metaheuristic algorithms:
  - **n_neighbors**: Number of nearest neighbors (1-50)
  - **weights**: Weight function ('uniform' or 'distance')
  - **metric**: Distance metric ('euclidean', 'manhattan', or 'cosine')
- Handle large-scale numerical data with proper preprocessing and transformation
- Export a production-ready model for deployment

## 🧠 Techniques & Algorithms Used

### 1. **Data Preprocessing**
- **Missing Value Imputation**: Median imputation for numerical features to handle missing data robustly
- **Data Cleaning**: Removal of commas from numerical strings and conversion to numeric types
- **Feature Selection**: Dropped irrelevant columns (SiriusXM Spins, Pandora Track Stations, etc.)
- **Categorical Handling**: Filled missing artist names with 'Unknown'

### 2. **Target Transformation**
- **Logarithmic Transformation** (`np.log1p`): Applied to the target variable (Spotify Streams) to:
  - Reduce the impact of extreme outliers
  - Normalize the distribution of streaming counts
  - Improve model performance and metric interpretability
  - Handle the wide range of streaming numbers (from thousands to billions)

### 3. **Feature Scaling**
- **MinMax Normalization**: Scales all features to [0, 1] range using `MinMaxScaler`
- Implemented within a scikit-learn Pipeline for consistent preprocessing

### 4. **Hyperparameter Optimization**

#### **Random Hill Climbing with Simulated Annealing**
This hybrid metaheuristic approach combines:

- **Random Hill Climbing**: 
  - Iteratively explores neighboring configurations
  - Accepts improvements unconditionally
  - Fast convergence to local optima

- **Simulated Annealing**:
  - Allows occasional acceptance of worse solutions to escape local optima
  - Acceptance probability: `P(accept) = exp(Δ/T)` where Δ is score difference and T is temperature
  - Temperature cooling schedule: `T = T × 0.95` per iteration
  - Initial temperature: 1000
  - Cooling rate: 0.95

**Hyperparameter Search Space**:
- `n_neighbors`: [1, 50] - Number of nearest neighbors
- `weights`: ['uniform', 'distance'] - Weight function for predictions
- `metric`: ['euclidean', 'manhattan', 'cosine'] - Distance metric

**Optimization Process**:
1. Generate 20 random initial configurations
2. Select the best as starting point
3. Run 200 iterations of hill climbing with simulated annealing
4. Track score evolution throughout optimization

### 5. **Model Evaluation**
- **Cross-Validation**: 5-fold CV with negative MSE scoring
- **Test Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)

### 6. **Visualization & Export**
- **Optimization History Plot**: Tracks MSE evolution across iterations
- **Model Serialization**: Exports trained pipeline using `joblib` for production use

## 📁 Project Structure

```
otimizacao/
├── Most Streamed Spotify Songs 2024.csv    # Dataset from Kaggle
├── otimizacao hiperparametros.py           # Main optimization script
├── best_knn_model.pkl                      # Trained model (serialized)
├── optimization_history.png                # Score evolution visualization
└── README.md                               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Running the Optimization

```bash
python "otimizacao hiperparametros.py"
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('best_knn_model.pkl')

# Prepare your data (same features as training)
# X_new = pd.DataFrame(...)

# Make predictions (returns log-transformed values)
predictions_log = model.predict(X_new)

# Convert back to original scale
predictions = np.expm1(predictions_log)
```

## 📊 Dataset

**Source**: [Most Streamed Spotify Songs 2024 - Kaggle](https://www.kaggle.com/)

**Features Include**:
- Spotify metrics (Streams, Playlist Count, Popularity)
- YouTube metrics (Views, Likes)
- TikTok metrics (Posts, Likes, Views)
- Apple Music, Pandora, Soundcloud, Shazam metrics
- And more...

**Target Variable**: `Spotify Streams`

## 📈 Results

The optimization process:
- Evaluates hundreds of hyperparameter configurations
- Uses cross-validation to prevent overfitting
- Produces a robust model with optimized performance
- Generates visualizations to track improvement

**Key Insights**:
- Log transformation significantly improves model stability
- Simulated annealing helps escape local optima
- Distance-weighted KNN often performs better for this task

## 🔧 Technical Stack

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning pipeline and algorithms
- **matplotlib**: Visualization
- **joblib**: Model serialization

## 📝 Key Learnings

1. **Metaheuristic Optimization**: Random Hill Climbing with Simulated Annealing provides a good balance between exploration and exploitation
2. **Target Transformation**: Log transformation is crucial for handling skewed distributions in streaming data
3. **Pipeline Architecture**: Using scikit-learn pipelines ensures consistent preprocessing in production
4. **Cross-Validation**: Essential for reliable hyperparameter selection

## 🎓 Academic Context

This project demonstrates:
- Advanced hyperparameter optimization techniques
- Proper handling of real-world messy data
- Application of metaheuristic algorithms to machine learning
- Best practices in model development and deployment

## 📄 License

This project is open-source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- Inspired by optimization techniques in machine learning research
- Built with scikit-learn and the Python data science ecosystem

---

**Author**: Developed as part of an AI/ML optimization study  
**Date**: December 2024
