import wfdb
import numpy as np
from src.features.label_extractor import LabelExtractor

metadata_path = "data/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
output_path = "dataqml/ecg_labels.csv"

def main():
    dl = LabelExtractor()
    dl.extract_label(metadata_path, output_path)

if __name__ == "__main__":
    main()