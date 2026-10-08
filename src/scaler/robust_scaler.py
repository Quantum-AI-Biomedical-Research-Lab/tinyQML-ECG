import numpy as np

class RobustScaler:

    def __init__(self):
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.q1 = np.percentile(x, 25, axis=0)
        self.q2 = np.percentile(x, 50, axis=0)
        self.q3 = np.percentile(x, 75, axis=0)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        iqr = self.q3 - self.q1
        iqr = np.where(iqr == 0, 1.0, iqr)
        return (x - self.q2) / iqr
    
    def fit_transform(self, x):
        return self.fit(x).transform(x)