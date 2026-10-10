import os
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from src.filters.pan_tompkins import PanTompkinsFilter
from src.features.data_extractor import DataExtractor, BeatFeature

FS = 100
LEAD = "II"
NUM_RECORDS = 5000
BASE = Path("data/physionet.org/files/ptb-xl/1.0.3")
OUT_FEATURES = Path("dataqml/ecg_features.csv")
OUT_LOG = Path("dataqml/extraction_log.csv")

def load_lead(filename_lr: str) -> np.ndarray:
    sig, meta = wfdb.rdsamp(str(BASE / filename_lr))
    return sig[:, meta["sig_name"].index(LEAD)].astype(np.float64)

def main():
    db = pd.read_csv(BASE / "ptbxl_database.csv", index_col="ecg_id").sort_index()
    db = db.head(NUM_RECORDS)
    de = DataExtractor(PanTompkinsFilter(fs=FS))
    rows, log = [], []
    for filename in db["filename_lr"]:
        record_name = Path(filename).name
        try:
            feats, res = de.extract_beat_features(record_name, load_lead(filename))
            rows.extend(feats)
            log.append({"record_id": record_name, "status": "ok",
                        "beats_detected": len(res.r_peaks), "beats_kept": len(feats),
                        "heart_rate_bpm": res.heart_rate_bpm})
        except Exception as e:
            log.append({"record_id": record_name, "status": f"error: {e}",
                        "beats_detected": 0, "beats_kept": 0, "heart_rate_bpm": np.nan})
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    de.features_to_dataframe(rows).to_csv(OUT_FEATURES, index=False)
    log_df = pd.DataFrame(log)
    log_df.to_csv(OUT_LOG, index=False)
    ok = log_df["status"] == "ok"
    print(f"Successful: {ok.sum()} / {len(log_df)} records, sum: {len(rows)} beats")
    print(f"Error: {(~ok).sum()} records")
    print(f"Records containing fewer than 3 beats: {(log_df['beats_kept'] < 3).sum()}")
    print(f"Beat rejection rate at the boundaries: "
          f"{1 - log_df['beats_kept'].sum() / max(log_df['beats_detected'].sum(), 1):.1%}")

if __name__ == "__main__":
    main()