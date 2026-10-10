import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from src.filters.pan_tompkins import PanTompkinsFilter, PanTompkinsFilterResult

@dataclass
class BeatFeature:
    record_id: str
    beat_index: int
    q_amplitude: float
    r_amplitude: float
    s_amplitude: float
    rr_interval_ms: float
    qrs_duration_ms: float

class DataExtractor:

    def __init__(self, pt: PanTompkinsFilter, baseline_ms: tuple[float, float] = (80.0, 20.0)):
        self.pt = pt
        self.base_start = round(baseline_ms[0] * 1e-3 * pt.fs)
        self.base_end = round(baseline_ms[1] * 1e-3 * pt.fs)

    def _baseline(self, ecg: np.ndarray, q: int) -> float | None:
        lo, hi = q - self.base_start, q - self.base_end
        if lo < 0 or hi <= lo:
            return None
        return float(np.median(ecg[lo:hi]))

    def extract_beat_features(self, record_id: str, ecg: np.ndarray) -> tuple[list[BeatFeature], PanTompkinsFilterResult]:
        ecg = np.asarray(ecg, dtype=np.float64)
        res = self.pt.run(ecg)
        q_pts, r_pts, s_pts = res.q_points, res.r_peaks, res.s_points
        n_beats = min(len(q_pts), len(r_pts), len(s_pts))
        margin = self.pt.qs_search
        fs = self.pt.fs
        features = []
        for i in range(n_beats):
            q, r, s = int(q_pts[i]), int(r_pts[i]), int(s_pts[i])
            if r - margin < 0 or r + margin >= len(ecg):
                continue
            base = self._baseline(ecg, q)
            if base is None:
                continue
            rr = (r - int(r_pts[i - 1])) / fs * 1e3 if i > 0 else np.nan
            features.append(BeatFeature(
                record_id=record_id,
                beat_index=i + 1,
                q_amplitude=float(ecg[q] - base),
                r_amplitude=float(ecg[r] - base),
                s_amplitude=float(ecg[s] - base),
                rr_interval_ms=float(rr),
                qrs_duration_ms=float((s - q) / fs * 1e3),
            ))
        return features, res
    
    def features_to_dataframe(self, features: list[BeatFeature]) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in features])

    def save_features(self, features: list[BeatFeature], path: str) -> None:
        df = self.features_to_dataframe(features)
        file_exists = os.path.exists(path)
        df.to_csv(path, mode="a", header=not file_exists, index=False)
        print(f"Saved {len(df)} beats to {path}")

    def extract_record(self, record_id: str, ecg: np.ndarray, path: str) -> None:
        features = self.extract_beat_features(record_id, ecg)
        self.save_features(features, path)