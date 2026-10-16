import argparse
import json
import os
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

BASE = "/opt/ml/processing"
TARGET_RECALL = 0.70
MIN_GROUP_POSITIVES = 100


def load_split(name):
    data = np.loadtxt(f"{BASE}/{name}/{name}.csv", delimiter=",")
    return data[:, 0].astype(int), xgb.DMatrix(data[:, 1:])


def metrics(y, prob, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def recall_gap(df, column):
    rows = {}
    for group, g in df.groupby(column):
        positives = int(g["y_true"].sum())
        if positives >= MIN_GROUP_POSITIVES:
            rows[str(group)] = float(recall_score(g["y_true"], g["y_pred"], zero_division=0))
    gap = max(rows.values()) - min(rows.values()) if len(rows) > 1 else 0.0
    return gap, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-pr-auc", type=float, default=0.20)
    parser.add_argument("--min-roc-auc", type=float, default=0.65)
    parser.add_argument("--min-precision", type=float, default=0.15)
    parser.add_argument("--max-recall-gap", type=float, default=0.10)
    args = parser.parse_args()

    with tarfile.open(f"{BASE}/model/model.tar.gz") as tar:
        tar.extractall(path=".")
    booster = xgb.Booster()
    booster.load_model("xgboost-model")

    # 1. Threshold tuning on validation (same as train-model.ipynb)
    y_val, d_val = load_split("validation")
    val_prob = booster.predict(d_val)
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    table = pd.DataFrame({"threshold": thr, "precision": prec[:-1], "recall": rec[:-1]})
    eligible = table[table["recall"] >= TARGET_RECALL]
    if len(eligible):
        threshold = float(eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]["threshold"])
    else:
        table["f1"] = 2 * table["precision"] * table["recall"] / (table["precision"] + table["recall"] + 1e-12)
        threshold = float(table.sort_values("f1", ascending=False).iloc[0]["threshold"])

    # 2. Test metrics
    y_test, d_test = load_split("test")
    test_prob = booster.predict(d_test)
    test_pred = (test_prob >= threshold).astype(int)
    test_m = metrics(y_test, test_prob, test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    # 3. Majority-class benchmark
    bench_pred = np.zeros(len(y_test), dtype=int)
    bench_m = metrics(y_test, bench_pred.astype(float), bench_pred)

    # 4. RecallGap fairness check
    sensitive = pd.read_csv(f"{BASE}/sensitive/test_sensitive.csv", keep_default_na=False)
    sensitive["y_true"], sensitive["y_pred"] = y_test, test_pred
    race_gap, race_recall = recall_gap(sensitive, "race")
    gender_gap, gender_recall = recall_gap(sensitive, "gender")
    worst_gap = max(race_gap, gender_gap)

    gates = [
        {"check": "Recall", "actual": test_m["recall"], "rule": f">= {args.min_recall}", "passed": test_m["recall"] >= args.min_recall},
        {"check": "PR-AUC", "actual": test_m["pr_auc"], "rule": f">= {args.min_pr_auc}", "passed": test_m["pr_auc"] >= args.min_pr_auc},
        {"check": "ROC-AUC", "actual": test_m["roc_auc"], "rule": f">= {args.min_roc_auc}", "passed": test_m["roc_auc"] >= args.min_roc_auc},
        {"check": "Precision", "actual": test_m["precision"], "rule": f">= {args.min_precision}", "passed": test_m["precision"] >= args.min_precision},
        {"check": "Beat benchmark PR-AUC", "actual": test_m["pr_auc"], "rule": f"> {bench_m['pr_auc']:.4f}", "passed": test_m["pr_auc"] > bench_m["pr_auc"]},
        {"check": "Beat benchmark recall", "actual": test_m["recall"], "rule": f"> {bench_m['recall']:.4f}", "passed": test_m["recall"] > bench_m["recall"]},
        {"check": "RecallGap (race, gender)", "actual": worst_gap, "rule": f"<= {args.max_recall_gap}", "passed": worst_gap <= args.max_recall_gap},
    ]
    gates_passed = int(all(g["passed"] for g in gates))

    report = {
        "binary_classification_metrics": {
            **{k: {"value": v, "standard_deviation": "NaN"} for k, v in test_m.items()},
            "confusion_matrix": {"0": {"0": int(tn), "1": int(fp)}, "1": {"0": int(fn), "1": int(tp)}},
        },
        "gates_passed": gates_passed,
        "decision_threshold": threshold,
        "quality_gates": [{**g, "passed": bool(g["passed"])} for g in gates],
        "benchmark_metrics": bench_m,
        "fairness": {
            "race_recall": race_recall, "race_gap": race_gap,
            "gender_recall": gender_recall, "gender_gap": gender_gap,
        },
    }

    os.makedirs(f"{BASE}/evaluation", exist_ok=True)
    with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
