import numpy as np

class MinMaxScaler:

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self._min = None
        self._max = None

    def fit(self, x):
        self._min = np.min(x, axis=0)
        self._max = np.max(x, axis=0)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        a, b = self.feature_range
        data_range = self._max - self._min
        data_range = np.where(data_range == 0, 1.0, data_range)
        return a + (x - self._min) * (b - a) / data_range

    def fit_transform(self, x):
        return self.fit(x).transform(x)