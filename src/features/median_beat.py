import numpy as np

class MedianBeat:

    def __init__(self, fs: float, pre_ms: float = 250.0, post_ms: float = 450.0, base_ms: float = 50.0):
        self.fs = float(fs)
        self.pre = round(pre_ms * 1e-3 * self.fs)
        self.post = round(post_ms * 1e-3 * self.fs)
        self.base = max(1, round(base_ms * 1e-3 * self.fs))

    @property
    def length(self) -> int:
        return self.pre + self.post

    def template(self, x: np.ndarray, r_peaks) -> np.ndarray | None:
        beats = [x[r - self.pre:r + self.post] for r in r_peaks if r - self.pre >= 0 and r + self.post <= len(x)]
        if len(beats) < 2:
            return None
        m = np.median(np.stack(beats), axis=0)
        return m - np.median(m[:self.base])