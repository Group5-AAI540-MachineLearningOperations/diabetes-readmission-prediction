import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "/opt/ml/processing"
RANDOM_STATE = 42
TARGET = "readmitted_30d"

CATEGORICAL = [
    "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]
NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
SENSITIVE = ["race", "gender", "age"]


def main():
    df = pd.read_csv(f"{BASE}/input/features.csv", keep_default_na=False)

    # Athena lowercases column names; restore the original names
    expected = CATEGORICAL + NUMERIC + SENSITIVE + [TARGET, "patient_nbr", "encounter_id"]
    df = df.rename(columns={c.lower(): c for c in expected})
    print("Feature Store rows:", df.shape)

    for c in CATEGORICAL + SENSITIVE:
        df[c] = df[c].astype(str).replace({"": "Unknown", "?": "Unknown"})
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c])
    df[TARGET] = df[TARGET].astype(int)

    # Patient-level stratified split (same as train-model.ipynb)
    patient_ids = df["patient_nbr"].astype(str)
    patient_table = (
        pd.DataFrame({"patient_nbr": patient_ids, "target": df[TARGET]})
        .groupby("patient_nbr", as_index=False)["target"].max()
    )
    train_p, temp_p = train_test_split(
        patient_table, test_size=0.30, random_state=RANDOM_STATE, stratify=patient_table["target"]
    )
    val_p, test_p = train_test_split(
        temp_p, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_p["target"]
    )
    masks = {
        "train": patient_ids.isin(set(train_p["patient_nbr"])),
        "validation": patient_ids.isin(set(val_p["patient_nbr"])),
        "test": patient_ids.isin(set(test_p["patient_nbr"])),
    }

    # Fit preprocessor on train only (same as train-model.ipynb)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL),
            ("numeric", StandardScaler(), NUMERIC),
        ],
        remainder="drop",
    )
    preprocessor.fit(df.loc[masks["train"], CATEGORICAL + NUMERIC])
    print("Transformed feature count:", len(preprocessor.get_feature_names_out()))

    for name, mask in masks.items():
        part = df.loc[mask]
        X_t = preprocessor.transform(part[CATEGORICAL + NUMERIC]).toarray()
        out = np.column_stack([part[TARGET].to_numpy(), X_t])
        os.makedirs(f"{BASE}/{name}", exist_ok=True)
        np.savetxt(f"{BASE}/{name}/{name}.csv", out, delimiter=",", fmt="%.6g")
        print(f"{name}: {part.shape[0]} rows, positive rate {part[TARGET].mean():.4f}")

    # Sensitive attributes for the RecallGap gate (same row order as test.csv)
    os.makedirs(f"{BASE}/sensitive", exist_ok=True)
    df.loc[masks["test"], SENSITIVE].to_csv(f"{BASE}/sensitive/test_sensitive.csv", index=False)

    # Save fitted preprocessor for deployment
    os.makedirs(f"{BASE}/preprocessor", exist_ok=True)
    joblib.dump(preprocessor, f"{BASE}/preprocessor/preprocessor.joblib")


if __name__ == "__main__":
    main()
