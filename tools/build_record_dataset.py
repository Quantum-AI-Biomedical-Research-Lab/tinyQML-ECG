import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from src.features.record_aggregator import META_COLS, aggregate_records

BEATS_CSV = Path("dataqml/dataset1.csv")
TEMPLATES = Path("dataqml/median/median_beats.npz")
PTBXL_DB = Path("data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv")
OUT = Path("dataqml/median/record_dataset.csv")
OUT_DIR = Path("dataqml/median")
DEPLOY = Path("deploy/artifacts")
N_PCA = 8
VAL_FOLD, TEST_FOLD = 9, 10

def is_train(df: pd.DataFrame) -> np.ndarray:
    return (~df["strat_fold"].isin([VAL_FOLD, TEST_FOLD])).to_numpy()

def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    db = pd.read_csv(PTBXL_DB, usecols=["filename_lr", "age", "sex"])
    db["record_id"] = db["filename_lr"].str.split("/").str[-1]
    db["age"] = db["age"].clip(upper=90)
    return df.merge(db[["record_id", "age", "sex"]], on="record_id", how="left")

def add_template_pca(df: pd.DataFrame) -> pd.DataFrame:
    d = np.load(TEMPLATES)
    temps = pd.DataFrame(d["templates"], index=d["record_ids"].astype(str))
    before = len(df)
    df = df[df["record_id"].isin(temps.index)].copy()
    print(f"Records have no median beat (removed): {before - len(df)}")
    T = temps.loc[df["record_id"]].to_numpy(dtype=np.float64)
    pca = PCA(n_components=N_PCA, random_state=0).fit(T[is_train(df)])
    Z = pca.transform(T)
    for k in range(N_PCA):
        df[f"beat_pca_{k + 1}"] = Z[:, k]
    print(f"PCA with {N_PCA} components retains {pca.explained_variance_ratio_.sum():.1%} of the variance")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "median_beat_pca.npz", mean=pca.mean_, components=pca.components_)
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in ["record_id", *META_COLS]]
    med = df.loc[is_train(df), feat_cols].median()
    missing = df[feat_cols].isna().mean()
    print("Highest missing rate:", missing.sort_values(ascending=False).head(5).round(3).to_dict())
    df[feat_cols] = df[feat_cols].fillna(med)
    (DEPLOY / "record_impute_medians.json").write_text(json.dumps(med.to_dict(), indent=4), encoding="utf-8")
    return df

def main():
    beats = pd.read_csv(BEATS_CSV, dtype={"record_id": str})
    df = aggregate_records(beats)
    df = add_demographics(df)
    df = add_template_pca(df)
    df = impute(df)
    df.to_csv(OUT, index=False)
    n_feat = len([c for c in df.columns if c not in ["record_id", *META_COLS]])
    print(f"Saved {OUT}: {len(df)} records, {n_feat} features, abnormality rate {df['label'].mean():.1%}")

if __name__ == "__main__":
    main()