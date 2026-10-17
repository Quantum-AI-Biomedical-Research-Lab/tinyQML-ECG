import csv
import threading
import time
from pathlib import Path

class InferenceLogger:

    def __init__(self, path: Path, feature_names: list):
        self.path = path
        self.fields = ["time", "status", "p", "hr", "n_beats", "latency_ms", "reasons",
                       "received", "dropped", *feature_names]
        self._lock = threading.Lock()
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
    
    def log(self, row: dict):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        row["reasons"] = ";".join(row.get("reasons", []))
        with self._lock, self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)