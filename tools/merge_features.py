import os
import pandas as pd

features = pd.read_csv("dataqml/ecg_features.csv")
labels = pd.read_csv("dataqml/ecg_labels.csv")
dataset1_path = "dataqml/dataset1.csv" 

def main():
    dataset = features.merge(labels, on="record_id", how="inner")
    dataset.to_csv(dataset1_path, index=False)

if __name__ == "__main__":
    main()