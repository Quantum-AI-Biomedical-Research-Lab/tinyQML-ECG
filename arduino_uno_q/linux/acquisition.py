import threading
import numpy as np

class EcgStream:

    def __init__(
        self,
        fs: float,
        window_s: float,
        adc_bits: int,
        vref: float,
        gain: float
    ):
        self.fs = fs
        self.n = int(round(window_s * fs))
        self.mv_per_count = vref / (2 ** adc_bits - 1) / gain * 1000.0
        self._x = np.zeros(self.n)
        self._lo = np.zeros(self.n, dtype=bool)
        self._filled = 0
        self._last_seq = None
        self.received_chunks = 0
        self.dropped_chunks = 0
        self._lock = threading.Lock()

    def on_chunk(self, payload: str):
        parts = str(payload).strip().split(",")
        if len(parts) < 3:
            return
        seq, leads_off = int(parts[0]), parts[1] == "1"
        vals = np.asarray(parts[2:], dtype=np.float64) * self.mv_per_count
        k = len(vals)
        with self._lock:
            if self._last_seq is not None and seq != self._last_seq + 1:
                self.dropped_chunks += max(seq - self._last_seq - 1, 1)
                self._filled = 0
            self._last_seq = seq
            self.received_chunks  += 1
            self._x = np.roll(self._x, -k)
            self._x[-k:] = vals
            self._lo = np.roll(self._lo, -k)
            self._lo[-k:] = leads_off
            self._filled = min(self.n, self._filled + k)
    
    def snapshot(self):
        with self._lock:
            return self._x.copy(), self._lo.copy(), self._filled >= self.n