import ast
import pandas as pd
from pathlib import Path

ABNORMAL_CODES = {
    "1AVB",      # First-degree atrioventricular block
    "2AVB",      # Second-degree atrioventricular block
    "3AVB",      # Third-degree / complete atrioventricular block
    "ABQRS",     # Abnormal QRS axis
    "AFIB",      # Atrial fibrillation
    "AFLT",      # Atrial flutter
    "ALMI",      # Anterolateral myocardial infarction
    "AMI",       # Acute myocardial infarction
    "ASMI",      # Anteroseptal myocardial infarction
    "ILMI",      # Inferolateral myocardial infarction
    "IMI",       # Inferior myocardial infarction
    "IPLMI",     # Inferoposterolateral myocardial infarction
    "IPMI",      # Inferoposterior myocardial infarction
    "LMI",       # Lateral myocardial infarction
    "PMI",       # Posterior myocardial infarction
    "ANEUR",     # Ventricular aneurysm
    "BIGU",      # Bigeminy
    "CLBBB",     # Complete left bundle branch block
    "CRBBB",     # Complete right bundle branch block
    "ILBBB",     # Incomplete left bundle branch block
    "IRBBB",     # Incomplete right bundle branch block
    "IVCD",      # Intraventricular conduction disturbance
    "DIG",       # Digitalis effect
    "EL",        # Electrolyte abnormality
    "HVOLT",     # High QRS voltage
    "LVOLT",     # Low QRS voltage
    "INJAL",     # Injury pattern in anterolateral leads
    "INJAS",     # Injury pattern in anteroseptal leads
    "INJIL",     # Injury pattern in inferolateral leads
    "INJIN",     # Injury pattern in inferior leads
    "INJLA",     # Injury pattern in lateral leads
    "INVT",      # Inverted T wave
    "ISCAL",     # Ischemic ST-T changes in anterolateral leads
    "ISCAN",     # Ischemic ST-T changes in anterior leads
    "ISCAS",     # Ischemic ST-T changes in anteroseptal leads
    "ISCIL",     # Ischemic ST-T changes in inferolateral leads
    "ISCIN",     # Ischemic ST-T changes in inferior leads
    "ISCLA",     # Ischemic ST-T changes in lateral leads
    "ISC_",      # Ischemic ST-T changes
    "LAFB",      # Left anterior fascicular block
    "LAO/LAE",   # Left axis deviation / left atrial enlargement
    "LNGQT",     # Long QT interval
    "LOWT",      # Low T-wave amplitude
    "LPFB",      # Left posterior fascicular block
    "LPR",       # Prolonged PR interval
    "LVH",       # Left ventricular hypertrophy
    "NDT",       # Nonspecific ST/T changes
    "NST_",      # Nonspecific ST changes
    "NT_",       # Nonspecific T-wave changes
    "PAC",       # Premature atrial contraction
    "PACE",      # Pacemaker rhythm
    "PRC(S)",    # Premature complex
    "PSVT",      # Paroxysmal supraventricular tachycardia
    "PVC",       # Premature ventricular contraction
    "QWAVE",     # Pathological Q wave
    "RAO/RAE",   # Right axis deviation / right atrial enlargement
    "RVH",       # Right ventricular hypertrophy
    "SARRH",     # Sinus arrhythmia
    "SBRAD",     # Sinus bradycardia
    "SEHYP",     # Septal hypertrophy
    "STACH",     # Sinus tachycardia
    "STD_",      # ST depression
    "STE_",      # ST elevation
    "SVARR",     # Supraventricular arrhythmia
    "SVTAC",     # Supraventricular tachycardia
    "TAB_",      # T-wave abnormality
    "TRIGU",     # Trigeminy
    "VCLVH",     # Voltage criteria for left ventricular hypertrophy
    "WPW",       # Wolff-Parkinson-White pattern
}

class LabelExtractor:

    def __init__(self):
        pass

    def parse_scp_codes(self, scp_codes_str: str) -> dict:
        if pd.isna(scp_codes_str): return {}
        return ast.literal_eval(scp_codes_str)

    def get_label(self, scp_codes: dict) -> int:
        if "NORM" in scp_codes and scp_codes["NORM"] > 0:
            return 0
        if any(code in scp_codes for code in ABNORMAL_CODES):
            return 1
        return -1

    def save_label(self, metadata_path: str) -> pd.DataFrame:
        metadata = pd.read_csv(metadata_path)
        rows = []
        for _, row in metadata.iterrows():
            scp_codes = self.parse_scp_codes(row["scp_codes"])
            label = self.get_label(scp_codes)
            record_id = Path(row["filename_lr"]).stem
            rows.append({
                "record_id": record_id,
                "label": label
            })
        return pd.DataFrame(rows)

    def extract_label(self, metadata_path: str, output_path: str) -> None:
        labels = self.save_label(metadata_path)
        labels.to_csv(output_path, index=False)
        print(f"Saved labels to: {output_path}")