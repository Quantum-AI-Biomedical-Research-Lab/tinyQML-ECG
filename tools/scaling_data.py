import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FEATURE_COLS = [
    "q_amplitude",
    "r_amplitude",
    "s_amplitude",
    "rr_interval_ms",
    "qrs_duration_ms",
]
META_COLS = ["record_id", "beat_index", "label", "patient_id", "strat_fold"]
QML_RANGE = (0.0, np.pi)
CLIP_PERCENTILES = (1.0, 99.0)

def drop_incomplete_beats(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[FEATURE_COLS].notna().all(axis=1)
    print(f"Removed {(~mask).sum()} beats with missing values (mostly the first beat due to missing RR intervals)")
    return df[mask].copy()

def split_by_fold(df: pd.DataFrame, val_fold: int = 9, test_fold: int = 10):
    train_df = df[~df["strat_fold"].isin([val_fold, test_fold])].copy()
    val_df = df[df["strat_fold"] == val_fold].copy()
    test_df = df[df["strat_fold"] == test_fold].copy()
    for name, part in (("Train", train_df), ("Val", val_df), ("Test", test_df)):
        rec = part.groupby("record_id")["label"].first()
        print(f"{name:5s}: {len(rec):5d} records, {len(part):6d} beats, abnormal {rec.mean():.1%}")
    leak = set(train_df["patient_id"]) & (set(val_df["patient_id"]) | set(test_df["patient_id"]))
    print(f"Patients appearing in both train and val/test sets: {len(leak)}")
    return train_df, val_df, test_df

def fit_scalers(X_train: np.ndarray):
    lo, hi = np.percentile(X_train, CLIP_PERCENTILES, axis=0)
    std_scaler = StandardScaler().fit(X_train)
    qml_scaler = MinMaxScaler(feature_range=QML_RANGE).fit(np.clip(X_train, lo, hi))
    return std_scaler, qml_scaler

def to_qml(qml_scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return qml_scaler.transform(np.clip(X, qml_scaler.data_min_, qml_scaler.data_max_))

def save_npz(path: Path, X_baseline: np.ndarray, X_qml: np.ndarray, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_baseline=X_baseline,
        X_qml=X_qml,
        y=df["label"].to_numpy(dtype=np.int64),
        record_ids=df["record_id"].to_numpy(dtype=str),
        beat_indices=df["beat_index"].to_numpy(dtype=np.int64),
        patient_ids=df["patient_id"].to_numpy(dtype=np.int64),
    )
    print(f"Saved {path}")

def save_scaled_csv(path: Path, df: pd.DataFrame, X_scaled: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[META_COLS].copy()
    out[FEATURE_COLS] = np.round(X_scaled, 6)
    out.to_csv(path, index=False)
    print(f"Saved {path}")

def export_scaler_json(scaler: Any, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"feature_names": FEATURE_COLS}
    if isinstance(scaler, StandardScaler):
        config.update(type="StandardScaler", mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    elif isinstance(scaler, MinMaxScaler):
        config.update(
            type="MinMaxScaler",
            feature_range=[float(v) for v in scaler.feature_range],
            min=scaler.min_.tolist(),
            scale=scaler.scale_.tolist(),
            clip_min=scaler.data_min_.tolist(),
            clip_max=scaler.data_max_.tolist(),
        )
    else:
        raise ValueError("This scaler type is not supported")
    json_path.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {json_path}")

def process_data(input_csv: Path, output_dir: Path, deploy_dir: Path, val_fold: int, test_fold: int):
    if not input_csv.exists():
        raise FileNotFoundError(f"{input_csv} not found")
    print(f"[1/6] Reading {input_csv}...")
    df = pd.read_csv(input_csv, dtype={"record_id": str})
    print(f"{len(df)} beats, {df['record_id'].nunique()} records")
    print("[2/6] Removing beats with missing values...")
    df = drop_incomplete_beats(df)
    print("[3/6] Splitting by strat_fold...")
    train_df, val_df, test_df = split_by_fold(df, val_fold, test_fold)
    X_tr, X_va, X_te = (part[FEATURE_COLS].to_numpy(dtype=np.float64) for part in (train_df, val_df, test_df))
    print("[4/6] Fitting scaler on the training set...")
    std_scaler, qml_scaler = fit_scalers(X_tr)
    std = [std_scaler.transform(X) for X in (X_tr, X_va, X_te)]
    qml = [to_qml(qml_scaler, X) for X in (X_tr, X_va, X_te)]
    for name, Xq in zip(("train", "val", "test"), qml):
        print(f"QML Angle {name}: [{Xq.min():.3f}, {Xq.max():.3f}]")
    print("[5/6] Saving the data...")
    for name, part, Xs, Xq in zip(("train", "val", "test"), (train_df, val_df, test_df), std, qml):
        save_npz(output_dir / f"{name}.npz", Xs, Xq, part)
    save_scaled_csv(output_dir / "train_baseline.csv", train_df, std[0])
    save_scaled_csv(output_dir / "train_qml.csv", train_df, qml[0])
    print("[6/6] Saving scaler for deployment...")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(std_scaler, deploy_dir / "std_scaler.pkl")
    joblib.dump(qml_scaler, deploy_dir / "qml_scaler.pkl")
    export_scaler_json(std_scaler, deploy_dir / "std_scaler.json")
    export_scaler_json(qml_scaler, deploy_dir / "qml_scaler.json")

def main():
    parser = argparse.ArgumentParser(description="Data Normalization and Splitting")
    parser.add_argument("--input", type=Path, default=Path("dataqml/dataset1.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("dataqml/scaled"))
    parser.add_argument("--deploydir", type=Path, default=Path("deploy/artifacts"))
    parser.add_argument("--valfold", type=int, default=9)
    parser.add_argument("--testfold", type=int, default=10)
    args = parser.parse_args()
    process_data(args.input, args.outdir, args.deploydir, args.valfold, args.testfold)

if __name__ == "__main__":
    main()