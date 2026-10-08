import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "q_amplitude",
    "r_amplitude",
    "s_amplitude",
    "rr_interval_ms",
    "qrs_duration_ms",
]

META_COLS = [
    "record_id",
    "beat_index",
    "label",
]

def clean_missing_values(
    df: pd.DataFrame,
    group_col: str = "record_id",
    target_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    df_clean = df.copy()
    if target_cols is None:
        target_cols = ["rr_interval_ms", "qrs_duration_ms", "q_amplitude", "s_amplitude"]
    for col in target_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby(group_col)[col].bfill()
            if df_clean[col].isna().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

def split_by_record(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    unique_records = df["record_id"].unique()
    train_records, temp_records = train_test_split(unique_records, test_size=test_size + val_size, random_state=random_state)
    relative_test_size = test_size / (test_size + val_size)
    val_records, test_records = train_test_split(temp_records, test_size=relative_test_size, random_state=random_state)
    train_df = df[df["record_id"].isin(train_records)].copy()
    val_df = df[df["record_id"].isin(val_records)].copy()
    test_df = df[df["record_id"].isin(test_records)].copy()
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Train beats: {len(train_df)}")
    print(f"Val beats: {len(val_df)}")
    print(f"Test beats: {len(test_df)}")
    return train_df, val_df, test_df

def fit_and_transform_scalers(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    std_scaler = StandardScaler()
    qml_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_std = std_scaler.fit_transform(X_train)
    X_train_qml = qml_scaler.fit_transform(X_train)
    X_val_std = std_scaler.transform(X_val)
    X_val_qml = qml_scaler.transform(X_val)
    X_test_std = std_scaler.transform(X_test)
    X_test_qml = qml_scaler.transform(X_test)
    return (X_train_std, X_val_std, X_test_std, X_train_qml, X_val_qml, X_test_qml, std_scaler, qml_scaler)

def save_npz(path: str, X_baseline, X_qml, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X_baseline=X_baseline,
        X_qml=X_qml,
        y=df["label"].to_numpy(dtype=np.int64),
        record_ids=df["record_id"].to_numpy(),
        beat_indices=df["beat_index"].to_numpy()
    )
    print(f"Saved dataset: {path}")

def save_scaled_csv(path: str, df: pd.DataFrame, X_scaled: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_export = df[META_COLS].copy()
    df_export[FEATURE_COLS] = np.round(X_scaled, 6)
    df_export.to_csv(path, index=False)
    print(f"Saved CSV: {path}")

def export_scaler_json(scaler: Any, feature_names: List[str], json_path: str):
    directory = os.path.dirname(json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    config: Dict[str, Any] = {"feature_names": feature_names}
    if isinstance(scaler, StandardScaler):
        config["type"] = "StandardScaler"
        config["mean"] = scaler.mean_.tolist()
        config["scale"] = scaler.scale_.tolist()
    elif isinstance(scaler, MinMaxScaler):
        config["type"] = "MinMaxScaler"
        config["feature_range"] = [
            float(scaler.feature_range[0]),
            float(scaler.feature_range[1]),
        ]
        config["min"] = scaler.min_.tolist()
        config["scale"] = scaler.scale_.tolist()
        config["data_min"] = scaler.data_min_.tolist()
        config["data_max"] = scaler.data_max_.tolist()
    else:
        raise ValueError("Unsupported scaler type.")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"Saved scaler config: {json_path}")

def process_data(
    input_csv: str = "dataqml/dataset1.csv",
    output_dir: str = "dataqml/scaled",
    deploy_dir: str = "deploy/artifacts",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    print(f"[1/7] Reading {input_csv}...")
    df_raw = pd.read_csv(input_csv)
    print(f"Total beats: {len(df_raw)}")
    print(f"Total records: {df_raw['record_id'].nunique()}")
    print(f"[2/7] Cleaning missing value...")
    df_clean = clean_missing_values(df_raw)
    print(f"[3/7] Splitting dataset into train-val-test set...")
    (train_df, val_df, test_df) = split_by_record(df_clean, test_size=test_size, val_size=val_size, random_state=random_state)
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    print(f"[4/7] Fitting scalers...")
    (X_train_std, X_val_std, X_test_std, X_train_qml, X_val_qml, X_test_qml, std_scaler, qml_scaler) = fit_and_transform_scalers(X_train, X_val, X_test)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    print(f"[5/7] Saving datasets...")
    save_npz(os.path.join(output_dir, "train.npz"), X_train_std, X_train_qml, train_df)
    save_npz(os.path.join(output_dir, "val.npz"), X_val_std, X_val_qml, val_df)
    save_npz(os.path.join(output_dir, "test.npz"), X_test_std, X_test_qml, test_df)
    print(f"[6/7] Saving CSV files...")
    save_scaled_csv(os.path.join(output_dir, "train_baseline.csv"), train_df, X_train_std)
    save_scaled_csv(os.path.join(output_dir, "train_qml.csv"), train_df, X_train_qml)
    print(f"[7/7] Saving scaler artifacts...")
    joblib.dump(std_scaler, os.path.join(deploy_dir, "std_scaler.pkl"))
    joblib.dump(qml_scaler, os.path.join(deploy_dir, "qml_scaler.pkl"))
    export_scaler_json(std_scaler, FEATURE_COLS, os.path.join(deploy_dir, "std_scaler.json"))
    export_scaler_json(qml_scaler, FEATURE_COLS, os.path.join(deploy_dir, "qml_scaler.json"))

def main():
    parser = argparse.ArgumentParser(description="Scaling data tool")
    parser.add_argument("--input", type=str, default="dataqml/dataset1.csv")
    parser.add_argument("--outdir", type=str, default="dataqml/scaled")
    parser.add_argument("--deploydir", type=str, default="deploy/artifacts")
    parser.add_argument("--testsize", type=float, default=0.2)
    parser.add_argument("--valsize", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_data(
        input_csv=args.input,
        output_dir=args.outdir,
        deploy_dir=args.deploydir,
        test_size=args.testsize,
        val_size=args.valsize,
        random_state=args.seed
    )

if __name__ == "__main__":
    main()