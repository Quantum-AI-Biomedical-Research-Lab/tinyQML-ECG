import numpy as np
import pandas as pd
import wfdb
import traceback
from pathlib import Path
from src.features.data_extractor import DataExtractor
from src.features.median_beat import MedianBeat
from src.filters.pan_tompkins import PanTompkinsFilter

FS = 100
LEAD = "II"
NUM_RECORDS = None
BASE = Path("data/physionet.org/files/ptb-xl/1.0.3")
OUT = Path("dataqml/median/median_beats.npz")

def load_lead(filename_lr: str) -> np.ndarray:
    sig, meta = wfdb.rdsamp(str(BASE / filename_lr))
    return sig[:, meta["sig_name"].index(LEAD)].astype(np.float64)

def main():
    db = pd.read_csv(BASE / "ptbxl_database.csv", index_col="ecg_id").sort_index()
    if NUM_RECORDS is not None:
        db = db.head(NUM_RECORDS)
    de = DataExtractor(PanTompkinsFilter(fs=FS))
    mb = MedianBeat(FS)
    ids, temps, failed = [], [], 0
    for filename in db["filename_lr"]:
        try:
            _, xf, _, beats = de.delineate_record(load_lead(filename))
            tpl = mb.template(xf, [b[2] for b in beats])
        except Exception:
            if failed == 0:
                traceback.print_exc()
            tpl = None
        if tpl is None:
            failed += 1
            continue
        ids.append(Path(filename).name)
        temps.append(tpl)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, record_ids=np.array(ids), templates=np.stack(temps).astype(np.float32), fs=FS, pre=mb.pre, post=mb.post)
    print(f"Saved {len(ids)}, median beat: {mb.length} samples, error {failed} records")

if __name__ == "__main__":
    main()