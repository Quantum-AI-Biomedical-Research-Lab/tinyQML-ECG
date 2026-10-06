import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass

@dataclass
class PanTompkinsFilterResult:
    stages: dict
    qrs_mwi: np.ndarray
    r_peaks: np.ndarray
    q_points: np.ndarray
    s_points: np.ndarray
    threshold_trace: np.ndarray
    qrs_duration_ms: float
    heart_rate_bpm: float

class PanTompkinsFilter:
    
    def __init__(
            self,
            fs: float = 500.0, 
            fc_low: float = 11.0,
            fc_high: float = 5.0,
            mwi_ms: float = 62.0,
            refractory_ms: float = 200.0,
            t_wave_ms: float = 360.0,
            qs_search_ms: float = 80.0,
            learn_s: float = 2.0,
            m_lp: int | None = None,
            m_hp: int | None = None
        ):
        self.fs = float(fs)
        self._f = np.linspace(1e-6, self.fs / 2, 8192)
        self.m_lp = m_lp or min(range(2, 128), key=lambda m: abs(self._lp_cutoff(m) - fc_low))
        self.m_hp = m_hp or min(range(4, 256, 2), key=lambda m: abs(self._hp_cutoff(m) - fc_high))
        self.gain_lp = self.m_lp ** 2
        self.gain_hp = self.m_hp
        self.delay_lp = self.m_lp - 1
        self.delay_hp = self.m_hp // 2
        self.delay_der = 2
        self.mwi_width = max(1, round(mwi_ms * 1e-3 * self.fs))
        self.fc_low = self._lp_cutoff(self.m_lp)
        self.fc_high = self._hp_cutoff(self.m_hp)
        self.refractory = round(refractory_ms * 1e-3 * self.fs)
        self.t_wave_window = round(t_wave_ms * 1e-3 * self.fs)
        self.qs_search = round(qs_search_ms * 1e-3 * self.fs)
        self.learn_len = round(learn_s * self.fs)

    def _wt(self, f: np.ndarray | None = None) -> np.ndarray:
        return 2 * np.pi * (self._f if f is None else f) / self.fs

    def _lp_magnitude(self, m: int | None = None, f: np.ndarray | None = None) -> np.ndarray:
        m = self.m_lp if m is None else m
        wt = self._wt(f)
        s = np.sin(m * wt / 2) / np.sin(wt / 2)
        return s ** 2 / m ** 2

    def _hp_magnitude(self, m: int | None = None, f: np.ndarray | None = None) -> np.ndarray:
        m = self.m_hp if m is None else m
        wt = self._wt(f)
        s = np.sin(m * wt / 2) / np.sin(wt / 2)
        return np.sqrt(np.maximum(m * m - 2 * m * s * np.cos(wt / 2) + s * s, 0.0)) / m
    
    def _der_magnitude(self, f: np.ndarray | None = None) -> np.ndarray:
        wt = self._wt(f)
        return self.fs / 4 * np.abs(np.sin(2 * wt) + 2 * np.sin(wt))

    def _lp_cutoff(self, m: int) -> float:
        return float(self._f[np.argmax(self._lp_magnitude(m) < (1.0 / np.sqrt(2.0)))])

    def _hp_cutoff(self, m: int) -> float:
        return float(self._f[np.argmax(self._hp_magnitude(m) < (1.0 / np.sqrt(2.0)))])

    def frequency_response(self, n: int = 2048):
        f = np.linspace(1e-6, self.fs / 2, n)
        lp, hp, der = self._lp_magnitude(f=f), self._hp_magnitude(f=f), self._der_magnitude(f=f)
        return f, {"lowpass": lp, "highpass": hp, "bandpass": lp * hp, "derivative": der / der.max()}

    def low_pass_filter(self, x: np.ndarray, steady_init: bool = True) -> np.ndarray:
        m = self.m_lp
        x = np.asarray(x, dtype=np.float64)
        x0 = x[0] if steady_init else 0.0
        xp = np.concatenate([np.full(2 * m, x0), x])
        y1 = y2 = self.gain_lp * x0
        y = np.empty(len(x))
        for n in range(len(x)):
            i = n + 2 * m
            yn = 2*y1 - y2 + xp[i] - 2*xp[i-m] + xp[i-2*m]
            y2, y1 = y1, yn
            y[n] = yn
        return y / self.gain_lp
    
    def high_pass_filter(self, x: np.ndarray, steady_init: bool = True) -> np.ndarray:
        m, h = self.m_hp, self.m_hp // 2
        x = np.asarray(x, dtype=np.float64)
        x0 = x[0] if steady_init else 0.0
        xp = np.concatenate([np.full(m, x0), x])
        y1 = 0.0
        y = np.empty(len(x))
        for n in range(len(x)):
            i = n + m
            y1 = y1 - xp[i] + m*xp[i-h] - m*xp[i-h-1] + xp[i-m]
            y[n] = y1
        return y / self.gain_hp

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        xp = np.concatenate([np.full(4, x[0]), x])
        return self.fs / 8 * (xp[4:] + 2*xp[3:3 + n] - 2*xp[1:1 + n] - xp[:n])

    @staticmethod
    def squaring(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) ** 2

    def moving_window_integration(self, x: np.ndarray) -> np.ndarray:
        w = self.mwi_width
        x = np.asarray(x, dtype=np.float64)
        xp = np.concatenate([np.full(w-1, x[0]), x])
        c = np.cumsum(np.concatenate([[0.0], xp]))
        return (c[w:] - c[:-w]) / w

    def filter_stages(self, x: np.ndarray) -> dict:
        lp = self.low_pass_filter(x)
        hp = self.high_pass_filter(lp)
        d = self.derivative(hp)
        sq = self.squaring(d)
        mwi = self.moving_window_integration(sq)
        return {"lowpass": lp, "highpass": hp, "derivative": d, "squared": sq, "mwi": mwi}

    def _adaptive_threshold(self, bp: np.ndarray, d: np.ndarray, mwi: np.ndarray):
        abs_bp = np.abs(bp)
        learn = max(1, min(len(mwi), self.learn_len))
        spki, npki = 0.25 * mwi[:learn].max(), 0.5 * mwi[:learn].mean()
        spkf, npkf = 0.25 * abs_bp[:learn].max(), 0.5 * abs_bp[:learn].mean()
        def thresholds():
            t1i = npki + 0.25 * (spki - npki)
            t1f = npkf + 0.25 * (spkf - npkf)
            return t1i, 0.5 * t1i, t1f, 0.5 * t1f
        peaks, _ = find_peaks(mwi)
        back = self.mwi_width + self.delay_der
        qrs, slopes, rr, pending, trace = [], [], [], [], []
        for p in peaks:
            if qrs and p - qrs[-1] <= self.refractory:
                continue
            lo = max(0, p - back)
            pki, pkf = mwi[p], abs_bp[lo: p+1].max()
            slope = np.abs(d[lo: p+1]).max()
            thr1i, thr2i, thr1f, thr2f = thresholds()
            if rr and pending and p - qrs[-1] > 1.66 * np.mean(rr[-8:]):
                cand = [c for c in pending if c[1] > thr2i and c[2] > thr2f]
                if cand:
                    c = max(cand, key=lambda item: item[1])
                    rr.append(c[0] - qrs[-1])
                    qrs.append(c[0])
                    slopes.append(c[3])
                    spki = 0.25 * c[1] + 0.75 * spki
                    spkf = 0.25 * c[2] + 0.75 * spkf
                    pending = [item for item in pending if item[0] > c[0]]
                    thr1i, thr2i, thr1f, thr2f = thresholds()
                    if p - qrs[-1] <= self.refractory:
                        continue
            is_t_wave = bool(qrs) and p - qrs[-1] < self.t_wave_window and slope < 0.5 * slopes[-1]
            if pki > thr1i and pkf > thr1f and not is_t_wave:
                if qrs:
                    rr.append(p - qrs[-1])
                qrs.append(p)
                slopes.append(slope)
                spki = 0.125 * pki + 0.875 * spki
                spkf = 0.125 * pkf + 0.875 * spkf
                pennding =[]
            else:
                npki = 0.125 * pki + 0.875 * npki
                npkf = 0.125 * pkf + 0.875 * npkf
                if not is_t_wave:
                    pending.append((p, pki, pkf, slope))
            trace.append((p, thresholds()[0]))
        return np.asarray(qrs, dtype=int), np.asarray(trace, dtype=np.float64).reshape(-1, 2)

    def _qrs_points(self, x: np.ndarray, qrs_mwi: np.ndarray):
        shift = self.delay_lp + self.delay_hp + self.delay_der + (self.mwi_width - 1) / 2
        haft = round(1.0 * self.fs)
        q_list, r_list, s_list = [], [], []
        for p in qrs_mwi:
            c = int(round(p - shift))
            lo, hi = max(0, c - half), min(len(x), c + hafe + 1)
            if hi - lo < 3:
                continue
            r = lo + int(np.argmax(x[lo:hi]))
            if r_list and r == r_list[-1]:
                continue
            ql, sh = max(0, r - self.qs_search), min(len(x), r + self.qs_search + 1)
            r_list.append(r)
            q_list.append(ql + int(np.argmin(x[ql:r + 1])))
            s_list.append(r + int(np.argmin(x[r:sh])))
        return (np.asarray(q_list, dtype=int), np.asarray(r_list, dtype=int), np.asarray(s_list, dtype=int))

    def run(self, x:np.ndarray) -> PanTompkinsFilterResult:
        x = np.asarray(x, dtype=np.float64)
        stages = self.filter_stages(x)
        qrs_mwi, trace = self._adaptive_threshold(stages["highpass"], stages["derivative"], stages["mwi"])
        q, r, s = self._qrs_points(x, qrs_mwi)
        qrs_ms = float(np.mean(s - q) / self.fs * 1e3) if len(r) else float("nan")
        hr = float(np.mean(60 * self.fs / np.diff(r))) if len(r) > 1 else float("nan")
        return PanTompkinsResult(stages, qrs_mwi, r, q, s, trace, qrs_ms, hr)

    def stage_delays(self) -> dict:
        bp = self.lp_delay + self.hp_delay
        der = bp + self.der_delay
        return {"lowpass": self.lp_delay, "highpass": bp, "derivative": der, "squared": der, "mwi": der + (self.mwi_width - 1) / 2}

    def summary_str(self) -> str:
        return(f"fs = {self.fs:g} Hz\n"
               f"Lowpass: M = {self.m_lp}, gain = {self.gain_lp}, delay = {self.delay_lp} samples, "
               f"fc = {self.fc_low:.1f} Hz\n"
               f"Highpass: M = {self.m_hp}, gain = {self.gain_hp}, delay = {self.delay_lp} samples,"
               f"fc = {self.fc_high:.1f} Hz\n"
               f"Derivative: delay = {self.delay_der} samples\n"
               f"Moving-Window Integration: {self.mwi_width} samples ({self.mwi_width / self.fs * 1e3:.0f} ms)")

    def summary(self):
        print(f"fs = {self.fs:g} Hz\n"
              f"Lowpass: M = {self.m_lp}, gain = {self.gain_lp}, delay = {self.delay_lp} samples, "
              f"fc = {self.fc_low:.1f} Hz\n"
              f"Highpass: M = {self.m_hp}, gain = {self.gain_hp}, delay = {self.delay_hp} samples,"
              f"fc = {self.fc_high:.1f} Hz\n"
              f"Derivative: delay = {self.delay_der} samples\n"
              f"Moving-Window Integration: {self.mwi_width} samples ({self.mwi_width / self.fs * 1e3:.0f} ms)")