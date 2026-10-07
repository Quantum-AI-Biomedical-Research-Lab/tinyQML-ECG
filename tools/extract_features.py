import wfdb
import numpy as np
from src.filters.pan_tompkins import PanTompkinsFilter
from src.features.data_extractor import DataExtractor, BeatFeature

FS = 100
output_path = "dataqml/ecg_features.csv"

def load_lead():
    path = "data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr"
    signal, meta = wfdb.rdsamp(path)
    return signal[:, 0].astype(np.float64)

def main():
    x = load_lead()
    pt = PanTompkinsFilter(fs=FS)
    de = DataExtractor(pt)
    de.extract_record("00001_lr", x, output_path)

if __name__ == "__main__":
    main()