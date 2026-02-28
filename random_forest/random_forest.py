import numpy as np
from .decision_tree import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=100, max_features=None):
        """
        Initialize the Random Forest Regressor.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the trees.
            max_features (int or float): The number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        # Build the random forest from the training set (X, y).
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            
            # Bootstrap sampling
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train the tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        # Create a bootstrap sample (random sampling with replacement).
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Aggregate predictions (mean)
        return np.mean(tree_preds, axis=0)
