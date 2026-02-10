# mlp_extractor.py

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class MLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layer_sizes=(128,), random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.2
        )

    def fit(self, X, y):
        self.mlp.fit(X, y)
        self.coefs_ = self.mlp.coefs_
        self.intercepts_ = self.mlp.intercepts_
        return self

    def transform(self, X):
        X_hidden = X @ self.coefs_[0] + self.intercepts_[0]
        return np.maximum(X_hidden, 0)
