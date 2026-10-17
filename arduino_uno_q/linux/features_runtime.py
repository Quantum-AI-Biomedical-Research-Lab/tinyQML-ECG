import json
import math
from pathlib import Path
import numpy as np
from src.features.data_extractor import DataExtractor
from src.features.median_beat import MedianBeat
from src.features.record_aggregator import aggregate_records
from src.filters.pan_tompkins import PanTompkinsFilter

class RecordFeatureRuntime:

    def __init__(
        self,
        artifacts: Path,
        fs: float,
        age: float,
        sex: int
    ):
        self.de = DataExtractor(PanTompkinsFilter(fs=fs))
        self.mb = MedianBeat(fs)
        pca = np.load(artifacts / "median_beat_pca.npz")
        self.pca_mean = pca["mean"]
        self.pca_comp = pca["components"]
        if len(self.pca_mean) != self.mb.length:
            raise ValueError(f"Median beat length {self.mb.length} does not match PCA length {len(self.pca_mean)}")
        self.medians = json.loads((artifacts / "record_impute_medians.json").read_text(encoding="utf-8"))
        self.age = float(age)
        self.sex = int(sex)

    def compute(self, x: np.ndarray):
        feats, res = self.de.extract_beat_features("live", x)
        hr = res.heart_rate_bpm
        info = {
            "n_detected": int(len(res.r_peaks)), 
            "n_beats": len(feats),
            "hr": None if hr is None or math.isnan(hr) else float(hr)}
        if len(feats) < 3:
            return None, info
        rec = aggregate_records(DataExtractor.features_to_dataframe(feats)).iloc[0].to_dict()
        _, xf, _, beats = self.de.delineate_record(x)
        tpl = self.mb.template(xf, [b[2] for b in beats])
        if tpl is None:
            return None, info
        z = (tpl - self.pca_mean) @ self.pca_comp.T
        for k, v in enumerate(z, start=1):
            rec[f"beat_pca_{k}"] = float(v)
        rec["age"], rec["sex"] = self.age, self.sex
        for key, med in self.medians.items():
            v = rec.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                rec[key] = med
        return rec, info