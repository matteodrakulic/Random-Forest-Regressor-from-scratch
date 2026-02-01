import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Represents a node in the decision tree.
        
        Args:
            feature_index (int): Index of the feature used for splitting (None for leaf).
            threshold (float): The value to split the feature at (None for leaf).
            left (Node): Left child node.
            right (Node): Right child node.
            value (float): The predicted value for the node (only for leaf nodes).
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2, max_features=None):
        """
        Initialize the Decision Tree Regressor.
        
        Args:
            min_samples_split (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the tree.
            max_features (int or float): The number of features to consider when looking for the best split.
                                         If float, then max_features is a percentage and int(max_features * n_features) are considered.
                                         If None, then max_features = n_features.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree from the training set (X, y).
        """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the tree.
        """
        num_samples, num_features = X.shape
        
        # Stopping criteria
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            # Find the best split
            best_split = self._get_best_split(X, y, num_features)
            
            # If information gain is positive, create sub-trees
            if best_split.get("var_red", 0) > 0:
                left_subtree = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_subtree = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree)
        
        # Leaf node
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _get_best_split(self, X, y, num_features):
        """
        Find the best split for the dataset.
        """
        best_split = {}
        max_var_red = -float("inf")
        
        # Determine the number of features to consider
        if self.max_features is not None:
            if isinstance(self.max_features, float):
                n_features_to_sample = int(self.max_features * num_features)
            else:
                n_features_to_sample = self.max_features
            
            # Ensure at least 1 feature is selected
            n_features_to_sample = max(1, min(n_features_to_sample, num_features))
            
            # Randomly select features
            feature_indices = np.random.choice(num_features, n_features_to_sample, replace=False)
        else:
            feature_indices = range(num_features)
        
        # Loop over selected features
        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            # Loop over all unique feature values present in the data
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                
                if len(X_left) > 0 and len(X_right) > 0:
                    curr_var_red = self._calculate_variance_reduction(y, y_left, y_right)
                    
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["X_left"] = X_left
                        best_split["y_left"] = y_left
                        best_split["X_right"] = X_right
                        best_split["y_right"] = y_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        return best_split

    def _split(self, X, y, feature_index, threshold):
        """
        Split the data based on a feature and a threshold.
        """
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        
        X_left = X[left_indices]
        y_left = y[left_indices]
        X_right = X[right_indices]
        y_right = y[right_indices]
        
        return X_left, y_left, X_right, y_right

    def _calculate_variance_reduction(self, y, y_left, y_right):
        """
        Calculate variance reduction.
        """
        var_total = np.var(y)
        var_left = np.var(y_left)
        var_right = np.var(y_right)
        
        frac_left = len(y_left) / len(y)
        frac_right = len(y_right) / len(y)
        
        # Formula: Var(parent) - Weighted_Average_Var(children)
        variance_reduction = var_total - (frac_left * var_left + frac_right * var_right)
        
        return variance_reduction

    def _calculate_leaf_value(self, y):
        """
        Calculate the value of a leaf node (mean of target values).
        """
        return np.mean(y)

    def predict(self, X):
        """
        Predict regression value for X.
        """
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def make_prediction(self, x, tree):
        """
        Predict a single data point.
        """
        if tree.value is not None:
            return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
