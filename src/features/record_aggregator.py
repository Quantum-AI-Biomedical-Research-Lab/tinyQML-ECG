import numpy as np
import pandas as pd

MORPH_COLS = [
    "q_amplitude",
    "r_amplitude",
    "s_amplitude",
    "p_amplitude",
    "t_amplitude",
    "st_level",
    "qrs_duration_ms",
    "pq_interval_ms",
    "qt_interval_ms",
    "qtc_ms",
    "st_segment_ms"
]

META_COLS = [
    "label",
    "patient_id",
    "strat_fold"
]

def _rmssd(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.sqrt(np.mean(np.diff(v) ** 2))) if len(v) > 2 else np.nan

def aggregate_records(beats: pd.DataFrame) -> pd.DataFrame:
    beats = beats.sort_values(["record_id", "beat_index"])
    g = beats.groupby("record_id")
    out = g[MORPH_COLS].median()
    rr = g["rr_interval_ms"]
    out["rr_mean_ms"] = rr.mean()
    out["rr_std_ms"] = rr.std()
    out["rr_rmssd_ms"] = rr.apply(lambda s: _rmssd(s.to_numpy(dtype=float)))
    out["rr_cv"] = out["rr_std_ms"] / out["rr_mean_ms"]
    out["heart_rate_bpm"] = 60000.0 / out["rr_mean_ms"]
    out["p_present_ratio"] = g["p_amplitude"].apply(lambda s: s.notna().mean())
    eps = 0.01
    out["qrs_pp_mv"] = out["r_amplitude"] - out["s_amplitude"]
    out["rs_ratio"] = out["r_amplitude"] / (out["s_amplitude"].abs() + eps)
    out["tr_ratio"] = out["t_amplitude"] / (out["r_amplitude"].abs() + eps)
    out["qr_ratio"] = out["q_amplitude"].abs() / (out["r_amplitude"].abs() + eps)
    out["n_beats"] = g.size()
    present = [c for c in META_COLS if c in beats.columns]
    return out.join(g[present].first()).reset_index()