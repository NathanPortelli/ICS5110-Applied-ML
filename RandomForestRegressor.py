from DecisionTreeRegressor import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as SKLearnDecisionTreeRegressor
import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, custom=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.custom = custom

    def fit(self, X, y, tree_params=None):
        if tree_params is None:
            tree_params = {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf
            }

        # Convert X and y to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        for _ in range(self.n_estimators):
            if self.custom:
                tree = DecisionTreeRegressor(**tree_params)
            else:
                tree = SKLearnDecisionTreeRegressor(**tree_params)

            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))

        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X.values)

        return np.mean(predictions, axis=1)
