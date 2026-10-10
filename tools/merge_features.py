from pathlib import Path
import pandas as pd

FEATURES = Path("dataqml/ecg_features.csv")
LABELS = Path("dataqml/ecg_labels.csv")
PTBXL_DB = Path("data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")
OUT = Path("dataqml/dataset1.csv")

def load_meta() -> pd.DataFrame:
    db = pd.read_csv(PTBXL_DB, usecols=["patient_id", "strat_fold", "filename_lr"])
    db["record_id"] = db["filename_lr"].str.split("/").str[-1]
    return db[["record_id", "patient_id", "strat_fold"]]

def main():
    feats = pd.read_csv(FEATURES)
    labels = pd.read_csv(LABELS)
    if labels["record_id"].duplicated().any():
        raise ValueError("Label file contains duplicate record_id values")
    f_ids, l_ids = set(feats["record_id"]), set(labels["record_id"])
    print(f"Features available but labels missing (excluded): {len(f_ids - l_ids)}")
    print(f"Labels available but features missing (excluded): {len(l_ids - f_ids)}")
    ds = feats.merge(labels, on="record_id", how="inner", validate="many_to_one")
    ds = ds.merge(load_meta(), on="record_id", how="left", validate="many_to_one")
    if ds["strat_fold"].isna().any():
        raise ValueError("Records not found in ptbxl_database.csv")
    ds["patient_id"] = ds["patient_id"].astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(OUT, index=False)
    rec = ds.groupby("record_id").first()
    print(f"\n{len(ds)} beats, {len(rec)} records, {rec['patient_id'].nunique()} patients, "
          f"abnormality Rate {rec['label'].mean():.1%}")
    print(rec.groupby("strat_fold")["label"].agg(num_records="size", abnormality_rate="mean").round(3))

if __name__ == "__main__":
    main()