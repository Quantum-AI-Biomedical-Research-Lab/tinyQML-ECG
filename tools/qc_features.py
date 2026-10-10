from pathlib import Path
import pandas as pd

FEATURES_CSV = Path("dataqml/ecg_features.csv")
LABELS_CSV = Path("dataqml/ecg_labels.csv")
PTBXL_DB = Path("data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")

def check_join(feat: pd.DataFrame, lab: pd.DataFrame):
    f_ids, l_ids = set(feat["record_id"]), set(lab["record_id"])
    print(f"Records with features but missing labels: {len(f_ids - l_ids)}")
    print(f"Records with labels but missing features: {len(l_ids - f_ids)}")
    print(f"Duplicate record_id values in the label file: {lab['record_id'].duplicated().sum()}")
    print(f"Abnormality rate: {lab['label'].mean():.1%} per {len(lab)} record")

def check_beats(feat: pd.DataFrame):
    n = feat.groupby("record_id").size()
    print(f"Number of beats per record: min {n.min()}, median {n.median():.0f}, max {n.max()}")
    print(f"Records with fewer than 3 beats: {(n < 3).sum()}")

def check_qrs(feat: pd.DataFrame):
    sd = feat.groupby("record_id")["qrs_duration_ms"].std()
    print(f"Median within-record QRS standard deviation: median {sd.median():.1f} ms")
    print(f"Records with QRS standard deviation above 20 ms: {(sd > 20).mean():.1%}")

def check_amplitudes(feat: pd.DataFrame):
    print(f"Beats with positive Q amplitude: {(feat['q_amplitude'] > 0).mean():.1%}")
    print(f"Beats with positive S amplitude: {(feat['s_amplitude'] > 0).mean():.1%}")
    print(f"Beats with negative R amplitude: {(feat['r_amplitude'] < 0).mean():.1%}")
    g = feat.groupby("record_id")["r_amplitude"]
    drift = (g.max() - g.min()) / g.mean().abs()
    print(f"Median within-record R amplitude variation (max − min) / mean: median {drift.median():.1%}")

def check_rr(feat: pd.DataFrame):
    rr = feat["rr_interval_ms"]
    first = feat.loc[feat["beat_index"] == 1, "rr_interval_ms"]
    print(f"Missing RR intervals: {rr.isna().mean():.1%}")
    print(f"RR intervals outside the 300–2000 ms range: {((rr < 300) | (rr > 2000)).mean():.1%}")
    print(f"First beats with valid RR intervals (should be 0%): {first.notna().mean():.1%}")

def check_patient_leak(lab: pd.DataFrame):
    if not PTBXL_DB.exists():
        print("ptbxl_database.csv not found; skipping patient leakage check")
        return
    db = pd.read_csv(PTBXL_DB, usecols=["patient_id", "filename_lr"])
    db["record_id"] = db["filename_lr"].str.split("/").str[-1]
    m = lab.merge(db, on="record_id", how="left")
    per_patient = m.groupby("patient_id").size()
    print(f"Patients with more than one record: {(per_patient > 1).sum()} / {len(per_patient)}")

def main():
    feat = pd.read_csv(FEATURES_CSV, dtype={"record_id": str})
    lab = pd.read_csv(LABELS_CSV, dtype={"record_id": str})
    checks = [
        ("Label Merging", check_join, (feat, lab)),
        ("Beat Count", check_beats, (feat,)),
        ("QRS", check_qrs, (feat,)),
        ("Amplitudes", check_amplitudes, (feat,)),
        ("RR Intervals", check_rr, (feat,)),
        ("Patient Leakage", check_patient_leak, (lab,)),
    ]
    for title, fn, args in checks:
        print(f"\n[{title}]")
        fn(*args)


if __name__ == "__main__":
    main()