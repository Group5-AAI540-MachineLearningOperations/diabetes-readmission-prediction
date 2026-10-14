import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

BASE = "/opt/ml/processing"

if __name__ == "__main__":
    with tarfile.open(f"{BASE}/model/model.tar.gz") as tar:
        tar.extractall(path=".")
    booster = xgb.Booster()
    booster.load_model("xgboost-model")

    df = pd.read_csv(f"{BASE}/test/test.csv", header=None)
    y_true = df.iloc[:, 0].to_numpy()
    X = xgb.DMatrix(df.iloc[:, 1:].to_numpy())

    y_prob = booster.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "binary_classification_metrics": {
            "auc": {"value": float(roc_auc_score(y_true, y_prob)), "standard_deviation": "NaN"},
            "accuracy": {"value": float(accuracy_score(y_true, y_pred)), "standard_deviation": "NaN"},
            "precision": {"value": float(precision_score(y_true, y_pred, zero_division=0)), "standard_deviation": "NaN"},
            "recall": {"value": float(recall_score(y_true, y_pred)), "standard_deviation": "NaN"},
            "f1": {"value": float(f1_score(y_true, y_pred)), "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(tn), "1": int(fp)},
                "1": {"0": int(fn), "1": int(tp)},
            },
        }
    }

    os.makedirs(f"{BASE}/evaluation", exist_ok=True)
    with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
