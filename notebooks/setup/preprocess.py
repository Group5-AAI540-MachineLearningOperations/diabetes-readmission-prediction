import os
import numpy as np
import pandas as pd

BASE = "/opt/ml/processing"
TARGET = "readmitted_30d"
DROP_COLS = [
    "encounter_id", "readmitted", "weight", "payer_code",
    "diag_1", "diag_2", "diag_3",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "examide", "citoglipton",
]

if __name__ == "__main__":
    df = pd.read_csv(f"{BASE}/input/diabetes_processed.csv")
    print("Input shape:", df.shape)

    # Patient-level split to prevent leakage
    rng = np.random.default_rng(42)
    patients = df["patient_nbr"].unique()
    rng.shuffle(patients)
    n = len(patients)
    train_p = set(patients[: int(0.70 * n)])
    val_p = set(patients[int(0.70 * n): int(0.85 * n)])
    split = df["patient_nbr"].map(
        lambda p: "train" if p in train_p else ("validation" if p in val_p else "test")
    )

    # Clean and encode
    df = df.drop(columns=DROP_COLS + ["patient_nbr"], errors="ignore")
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].replace("?", "Unknown").fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)

    # Label first for built-in XGBoost
    df = df[[TARGET] + [c for c in df.columns if c != TARGET]]

    for name in ["train", "validation", "test"]:
        part = df[split == name]
        out_dir = f"{BASE}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        part.to_csv(f"{out_dir}/{name}.csv", header=False, index=False)
        print(f"{name}: {part.shape}, positive rate {part[TARGET].mean():.3f}")
