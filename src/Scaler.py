import numpy as np

class Scaler:
    '''Map to 0-1 value, and also do its inverse'''
    
    def __init__(self):
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        X = np.array(X)
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.data_min) / (self.data_max - self.data_min)

    def inverse_transform(self, X):
        X = np.array(X)
        return X * (self.data_max - self.data_min) + self.data_min
