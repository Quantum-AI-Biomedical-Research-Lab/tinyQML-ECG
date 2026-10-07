import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
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

    def __init__(self, pt: PanTompkinsFilter):
        self.pt = pt

    def extract_beat_features(self, record_id: str, ecg: np.ndarray) -> list[BeatFeature]:
        result = self.pt.run(ecg)
        q_points = np.asarray(result.q_points)
        r_peaks = np.asarray(result.r_peaks)
        s_points = np.asarray(result.s_points)
        n_beats = min(len(q_points), len(r_peaks), len(s_points))
        features = []
        for i in range(n_beats):
            q = int(q_points[i])
            r = int(r_peaks[i])
            s = int(s_points[i])
            q_amp = float(ecg[q])
            r_amp = float(ecg[r])
            s_amp = float(ecg[s])
            qrs_duration_ms = float((s - q) / self.pt.fs * 1e3)
            if i == 0: rr_interval_ms = np.nan
            else: rr_interval_ms = float((r_peaks[i] - r_peaks[i - 1]) / self.pt.fs * 1e3)
            features.append(
                BeatFeature(
                    record_id=record_id,
                    beat_index=i+1,
                    q_amplitude=q_amp,
                    r_amplitude=r_amp,
                    s_amplitude=s_amp,
                    rr_interval_ms=rr_interval_ms,
                    qrs_duration_ms=qrs_duration_ms
                )
            )
        if len(features) >= 2:
            if features[0].rr_interval_ms is None or np.isnan(features[0].rr_interval_ms):
                features[0].rr_interval_ms = features[1].rr_interval_ms
        return features

    def features_to_dataframe(self, features: list[BeatFeature]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "record_id": f.record_id,
                    "beat_index": f.beat_index,
                    "q_amplitude": f.q_amplitude,
                    "r_amplitude": f.r_amplitude,
                    "s_amplitude": f.s_amplitude,
                    "rr_interval_ms": f.rr_interval_ms,
                    "qrs_duration_ms": f.qrs_duration_ms
                }
                for f in features
            ]
        )

    def save_features(self, features: list[BeatFeature], path: str) -> None:
        df = self.features_to_dataframe(features)
        file_exists = os.path.exists(path)
        df.to_csv(path, mode="a", header=not file_exists, index=False)
        print(f"Saved {len(df)} beats to {path}")

    def extract_record(self, record_id: str, ecg: np.ndarray, path: str) -> None:
        features = self.extract_beat_features(record_id, ecg)
        self.save_features(features, path)