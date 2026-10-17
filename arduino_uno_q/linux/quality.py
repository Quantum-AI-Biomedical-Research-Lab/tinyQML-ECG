import numpy as np
from dataclasses import dataclass, field
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

@dataclass
class Quality:
    ok: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

class SignalQuality:

    def __init__(self, fs: float, cfg: dict):
        self.cfg = cfg
        self.sos = butter(2, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")

    def check_signal(self, x: np.ndarray, leads_off: np.ndarray) -> Quality:
        c = self.cfg
        xf = sosfiltfilt(self.sos, x)
        span = x.max() - x.min()
        tol = 1e-6 * max(span, 1e-9)
        sat = float(np.mean((x >= x.max() - tol) | (x <= x.min() + tol)))
        m = {
            "leads_off_frac": float(leads_off.mean()),
            "std_mv": float(xf.std()),
            "kurtosis": float(kurtosis(xf, fisher=False)),
            "saturation_frac": sat,
        }
        reasons = []
        if m["leads_off_frac"] > c["max_leads_off_frac"]:
            reasons.append("leads_off")
        if m["std_mv"] < c["min_std_mv"]:
            reasons.append("flatline")
        if m["std_mv"] > c["max_std_mv"]:
            reasons.append("excessive_noise")
        if m["saturation_frac"] > c["max_saturation_frac"]:
            reasons.append("saturation")
        if m["kurtosis"] < c["min_kurtosis"]:
            reasons.append("low_kurtosis")
        return Quality(not reasons, reasons, m)
    
    def check_beats(self, info: dict) -> list:
        reasons = []
        if info.get("n_beats", 0) < self.cfg["min_beats"]:
            reasons.append("too_few_beats")
        hr = info.get("hr")
        lo, hi = self.cfg["hr_range"]
        if hr is None or not (lo <= hr <= hi):
            reasons.append("implausible_heart_rate")
        return reasons