import numpy as np
from dataclasses import dataclass

@dataclass
class WaveFeatures:
    baseline: float
    p_onset: int | None
    p_peak: int | None
    t_onset: int | None
    t_peak: int | None
    t_end: int | None
    p_amplitude: float
    t_amplitude: float
    st_level: float
    pq_interval_ms: float
    qt_interval_ms: float
    qtc_ms: float
    st_segment_ms: float

class WaveDelineator:

    def __init__(
        self,
        fs: float,
        p_search_ms: tuple[float, float] = (300.0, 40.0),
        base_ms: tuple[float, float] = (40.0, 10.0),
        j_offset_ms: float = 20.0,
        st_point_ms: float = 60.0,
        t_search_ms: tuple[float, float] = (100.0, 450.0),
        p_min_mv: float = 0.03,
        flat_ratio: float = 0.1
    ):
        self.fs = float(fs)
        to_n = lambda v: max(1, round(v * 1e-3 * self.fs))
        self.p_lo, self.p_hi = to_n(p_search_ms[0]), to_n(p_search_ms[1])
        self.base_lo, self.base_hi = to_n(base_ms[0]), to_n(base_ms[1])
        self.j_offset = to_n(j_offset_ms)
        self.st_point = to_n(st_point_ms)
        self.t_lo, self.t_hi = to_n(t_search_ms[0]), to_n(t_search_ms[1])
        self.t_end_margin = to_n(100.0)
        self.p_min = p_min_mv
        self.flat_ratio = flat_ratio

    def _ms(self, n: float) -> float:
        return float(n / self.fs * 1e3)

    def _baseline(self, x: np.ndarray, q: int) -> float:
        lo, hi = q - self.base_lo, q - self.base_hi
        if lo < 0 or hi <= lo:
            return float("nan")
        return float(np.median(x[lo:hi]))

    def _find_p(self, x: np.ndarray, d: np.ndarray, q: int, base: float):
        lo, hi = q - self.p_lo, q - self.p_hi
        if lo < 0 or hi - lo < 3:
            return None, None, float("nan")
        seg = x[lo:hi] - base
        k = int(np.argmax(np.abs(seg)))
        amp = float(seg[k])
        if abs(amp) < self.p_min:
            return None, None, float("nan")
        peak = lo + k
        sign = np.sign(amp)
        rising = sign * d[lo:peak + 1]
        if rising.max() <= 0:
            return lo, peak, amp
        i_max = lo + int(np.argmax(rising))
        thr = self.flat_ratio * rising.max()
        onset = lo
        for i in range(i_max, lo - 1, -1):
            if sign * d[i] <= thr:
                onset = i
                break
        return onset, peak, amp

    def _tangent_to_baseline(self, x: np.ndarray, d: np.ndarray, k: int, base: float) -> float | None:
        if d[k] == 0:
            return None
        return k + (base - x[k]) / d[k] * self.fs

    def _find_t(self, x: np.ndarray, d: np.ndarray, r: int, s: int, rr_n: float, base: float):
        lo = max(r + self.t_lo, s + self.j_offset + 1)
        hi = r + self.t_hi
        if not np.isnan(rr_n):
            hi = min(hi, r + int(0.7 * rr_n))
        hi = min(hi, len(x))
        if hi - lo < 4:
            return None, None, None, float("nan")
        seg = x[lo:hi] - base
        k = int(np.argmax(np.abs(seg)))
        peak, amp = lo + k, float(seg[k])
        sign = np.sign(amp) if amp != 0 else 1.0
        end = None
        if hi - peak >= 2:
            k_desc = peak + int(np.argmin(sign * d[peak:hi]))
            e = self._tangent_to_baseline(x, d, k_desc, base)
            if e is not None:
                end = int(round(np.clip(e, k_desc, min(len(x) - 1, hi + self.t_end_margin))))
        onset = None
        if peak - lo >= 2:
            k_asc = lo + int(np.argmax(sign * d[lo:peak + 1]))
            o = self._tangent_to_baseline(x, d, k_asc, base)
            if o is not None:
                onset = int(round(np.clip(o, s + self.j_offset, k_asc)))
        return onset, peak, end, amp

    def run(self, x: np.ndarray, d: np.ndarray, q: int, r: int, s: int, rr_ms: float) -> WaveFeatures:
        nan = float("nan")
        base = self._baseline(x, q)
        if np.isnan(base):
            return WaveFeatures(nan, None, None, None, None, None, nan, nan, nan, nan, nan, nan, nan)
        j = s + self.j_offset
        st_idx = j + self.st_point
        st_level = float(x[st_idx] - base) if st_idx < len(x) else nan
        p_on, p_pk, p_amp = self._find_p(x, d, q, base)
        pq = self._ms(q - p_on) if p_on is not None else nan
        rr_n = rr_ms * 1e-3 * self.fs if not np.isnan(rr_ms) else nan
        t_on, t_pk, t_end, t_amp = self._find_t(x, d, r, s, rr_n, base)
        qt = self._ms(t_end - q) if t_end is not None else nan
        qtc = qt / np.sqrt(rr_ms * 1e-3) if not (np.isnan(qt) or np.isnan(rr_ms)) else nan
        st_seg = self._ms(t_on - j) if t_on is not None and t_on > j else nan
        return WaveFeatures(base, p_on, p_pk, t_on, t_pk, t_end, p_amp, t_amp, st_level, pq, qt, qtc, st_seg)