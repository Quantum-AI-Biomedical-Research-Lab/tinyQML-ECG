import pandas as pd

features = pd.read_csv("dataqml/ecg_features.csv")
labels = pd.read_csv("dataqml/ecg_labels.csv")

def main():
    dataset = features.merge(labels, on="record_id", how="inner")
    dataset.to_csv("dataqml/dataset1.csv", index=False)

if __name__ == "__main__":
    main()