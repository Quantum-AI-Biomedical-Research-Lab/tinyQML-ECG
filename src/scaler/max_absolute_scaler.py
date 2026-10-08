import numpy as np

class MaxAbsoluteScaler:

    def __init__(self):
        self.max_abs = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.max_abs = np.max(np.abs(x), axis=0)
        self.max_abs = np.where(self.max_abs == 0, 1.0, self.max_abs)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return x / self.max_abs
    
    def fit_transform(self, x):
        return self.fit(x).transform(x)