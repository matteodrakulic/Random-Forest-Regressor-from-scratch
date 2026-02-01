# Random Forest Regressor from Scratch

A Python implementation of a Random Forest Regressor built from scratch using only `numpy`. This project demonstrates the inner workings of decision trees and ensemble learning.

## Features

*   **Decision Tree Regressor**:
    *   Uses Variance Reduction as the splitting criterion.
    *   Supports maximum depth and minimum samples split constraints.
*   **Random Forest Regressor**:
    *   Implements Bagging (Bootstrap Aggregating).
    *   Supports Feature Subsampling (`max_features`) for decorrelating trees.
    *   Aggregates predictions via averaging.
*   **Zero Dependencies**: The core logic relies solely on `numpy`.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd random-forest-scratch
```

Install dependencies (only `numpy` is required for the model):

```bash
pip install numpy
```

## Usage

### Training a Model

```python
import numpy as np
from random_forest.random_forest import RandomForestRegressor

# Generate dummy data
X = np.random.rand(100, 5)
y = np.sum(X, axis=1)

# Initialize and train
rf = RandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=10, max_features=0.8)
rf.fit(X, y)

# Predict
predictions = rf.predict(X[:5])
print(predictions)
```

## Testing

To run the test script which trains the model on a synthetic dataset and evaluates performance:

```bash
python3 tests/test_random_forest.py
```

## Project Structure

*   `random_forest/`: Core library package.
    *   `decision_tree.py`: Implementation of a single Decision Tree.
    *   `random_forest.py`: Implementation of the Random Forest ensemble.
*   `tests/`: Test scripts.
    *   `test_random_forest.py`: Performance evaluation script.

