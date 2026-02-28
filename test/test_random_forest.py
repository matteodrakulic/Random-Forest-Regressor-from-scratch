import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path to resolve the random_forest package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from random_forest.random_forest import RandomForestRegressor
import time

def generate_synthetic_data(n_samples=500, n_features=10, noise=0.1):
    """
    Generate synthetic regression data.
    y = x_0 + 2*x_1 - 3*x_2 + noise
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    
    # Only the first 3 features are informative
    y = X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] + 0.5 * X[:, 1] * X[:, 2]
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    return X, y

def train_test_split(X, y, test_size=0.2):
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression: True vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig('rf_results.png')
    print("Plot saved as rf_results.png")

def test_random_forest():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=500, n_features=10)
    
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Data shape: X_train={X_train.shape}, X_test={X_test.shape}")
    
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=20, min_samples_split=2, max_depth=10, max_features=0.8)
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.4f} seconds.")
    
    print("\nMaking predictions...")
    y_pred = rf.predict(X_test)
    
    print("\nEvaluation:")
    print(f"MSE: {mse(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mse(y_test, y_pred)):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    test_random_forest()
