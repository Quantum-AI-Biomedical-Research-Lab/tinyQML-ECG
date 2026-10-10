import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from dataclasses import asdict, dataclass
from src.filters.delineation import WaveDelineator, WaveFeatures
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
    p_amplitude: float
    t_amplitude: float
    st_level: float
    pq_interval_ms: float
    qt_interval_ms: float
    qtc_ms: float
    st_segment_ms: float

class DataExtractor:

    def __init__(self, pt: PanTompkinsFilter, band_hz: tuple[float, float] = (0.5, 40.0)):
        self.pt = pt
        self.fs = pt.fs
        self.sos = butter(2, band_hz, btype="bandpass", fs=self.fs, output="sos")
        self.delineator = WaveDelineator(self.fs)

    def delineate_record(self, ecg: np.ndarray):
        ecg = np.asarray(ecg, dtype=np.float64)
        res = self.pt.run(ecg)
        xf = sosfiltfilt(self.sos, ecg)
        d = np.gradient(xf) * self.fs
        q_pts, r_pts, s_pts = res.q_points, res.r_peaks, res.s_points
        n = min(len(q_pts), len(r_pts), len(s_pts))
        margin = self.pt.qs_search
        beats = []
        for i in range(n):
            q, r, s = int(q_pts[i]), int(r_pts[i]), int(s_pts[i])
            if r - margin < 0 or r + margin >= len(xf): continue
            rr = (r - int(r_pts[i - 1])) / self.fs * 1e3 if i > 0 else float("nan")
            w = self.delineator.run(xf, d, q, r, s, rr)
            if np.isnan(w.baseline): continue
            beats.append((i, q, r, s, rr, w))
        return res, xf, beats

    def extract_beat_features(self, record_id: str, ecg: np.ndarray) -> tuple[list[BeatFeature], PanTompkinsFilterResult]:
        res, xf, beats = self.delineate_record(ecg)
        features = []
        for i, q, r, s, rr, w in beats:
            features.append(BeatFeature(
                record_id=record_id,
                beat_index=i + 1,
                q_amplitude=float(xf[q] - w.baseline),
                r_amplitude=float(xf[r] - w.baseline),
                s_amplitude=float(xf[s] - w.baseline),
                rr_interval_ms=float(rr),
                qrs_duration_ms=float((s - q) / self.fs * 1e3),
                p_amplitude=w.p_amplitude,
                t_amplitude=w.t_amplitude,
                st_level=w.st_level,
                pq_interval_ms=w.pq_interval_ms,
                qt_interval_ms=w.qt_interval_ms,
                qtc_ms=w.qtc_ms,
                st_segment_ms=w.st_segment_ms,
            ))
        return features, res

    @staticmethod
    def features_to_dataframe(features: list[BeatFeature]) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in features])