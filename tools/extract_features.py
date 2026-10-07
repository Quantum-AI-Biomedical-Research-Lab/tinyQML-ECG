import os
import wfdb
import numpy as np
from src.filters.pan_tompkins import PanTompkinsFilter
from src.features.data_extractor import DataExtractor, BeatFeature

FS = 100
NUM_RECORD = 1000
base_path = "data/physionet.org/files/ptb-xl/1.0.3/records100"
output_path = "dataqml/ecg_features.csv"

def load_lead(record_id):
    record = f"{record_id:05d}"
    folder = f"{(record_id // 1000) * 1000:05d}"
    path = os.path.join(base_path, folder, f"{record}_lr")
    if not os.path.exists(path + ".hea"):
        return None
    signal, meta = wfdb.rdsamp(path)
    return signal[:, 0].astype(np.float64)

def main():
    pt = PanTompkinsFilter(fs=FS)
    de = DataExtractor(pt)
    skipped = []
    if os.path.exists(output_path):
        os.remove(output_path)
    for record_id in range(1, NUM_RECORD):
        record = f"{record_id:05d}"
        record_name = f"{record}_lr"
        try:
            x = load_lead(record_id)
            if x is None:
                print(f"SKIP: {record}_lr not found")
                skipped.append(record)
                continue
            de.extract_record(record_name, x, output_path)
        except Exception as e:
            print(f"ERROR {record}_lr: {e}")
            skipped.append(record)
    if skipped:
        print("\nSkipped records:")
        print(", ".join(skipped))

if __name__ == "__main__":
    main()