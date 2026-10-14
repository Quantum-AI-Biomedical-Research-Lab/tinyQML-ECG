import json
from dataclasses import dataclass
import numpy as np

@dataclass
class AngleScaler:
    lo: np.ndarray
    hi: np.ndarray
    max_angle: float = float(np.pi)

    @classmethod
    def fit(cls, X: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0),
            max_angle: float = float(np.pi)) -> "AngleScaler":
        lo, hi = np.percentile(np.asarray(X, dtype=np.float64), percentiles, axis=0)
        hi = np.where(hi > lo, hi, lo + 1e-9)
        return cls(lo, hi, max_angle)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=np.float64), self.lo, self.hi)
        return self.max_angle * (X - self.lo) / (self.hi - self.lo)

    def to_json(self) -> str:
        return json.dumps({"lo": self.lo.tolist(), "hi": self.hi.tolist(),
                           "max_angle": self.max_angle}, indent=4)

    @classmethod
    def from_json(cls, text: str) -> "AngleScaler":
        d = json.loads(text)
        return cls(np.asarray(d["lo"]), np.asarray(d["hi"]), float(d["max_angle"]))