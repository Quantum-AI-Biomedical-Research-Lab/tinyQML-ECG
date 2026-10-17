import json
from pathlib import Path
import numpy as np
import pandas as pd
from arduino_uno_q.linux.vqc_runtime import VQCRuntime

rt = VQCRuntime(Path("arduino_uno_q/linux/artifacts"))
df = pd.read_csv("dataqml/median/record_dataset.csv", dtype={"record_id": str})
ref = pd.read_csv("results/quantum/qml_test_predictions.csv", dtype={"record_id": str})
test = df.set_index("record_id").loc[ref["record_id"]]
p = np.array([rt.predict(row[rt.features].to_dict())[0] for _, row in test.iterrows()])
print(f"Maximum deviation from PyTorch: {np.max(np.abs(p - ref['p'].to_numpy())):.2e}")