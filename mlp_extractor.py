import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layer_sizes=(128,), random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
