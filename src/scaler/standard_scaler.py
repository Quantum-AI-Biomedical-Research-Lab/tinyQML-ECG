import numpy as np

class :

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x):
        return self.fit(x).transform(x)

class ParetoScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / np.sqrt(self.std)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

class VASTScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.mean) / self.std) * (self.mean / self.std)

    def fit_transform(self, x):
        return self.fit(x).transform(x)