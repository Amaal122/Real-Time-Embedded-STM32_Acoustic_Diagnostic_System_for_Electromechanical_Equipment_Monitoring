#!/usr/bin/env python3
"""One-time evaluation of a frozen XGBoost model on a held-out subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("xgboost_comparison/train_val_test/tuned_model"))
    parser.add_argument("--test-subject", default="id_06")
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/train_val_test/tuned_model/test_evaluation"))
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if args.test_subject in metadata.get("fit_subjects", []):
        raise ValueError("Refusing evaluation: the requested test subject was used to fit the model")
    test = pd.read_csv(args.input)
    test = test[test.subject == args.test_subject].copy()
    if test.empty:
        raise ValueError(f"No rows found for held-out subject {args.test_subject}")
    X = test[metadata["selected_features"]].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = (test.label.to_numpy() == "abnormal").astype(np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("Test subject must have both normal and abnormal examples")

    model = XGBClassifier()
    model.load_model(args.model_dir / metadata["model_file"])
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(metadata["prediction_threshold"])
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    result = {
        "test_subject": args.test_subject, "n_windows": len(test),
        "threshold_from_validation": threshold,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_abnormal": float(precision_score(y, predictions, zero_division=0)),
        "recall_abnormal": float(recall_score(y, predictions, zero_division=0)),
        "f1_abnormal": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "model_dir": str(args.model_dir),
        "interpretation_note": "One held-out subject; small subject count and prior exploratory use mean this is retrospective evidence, not a fully independent estimate.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "subject": test.subject.to_numpy(), "file": test.file.to_numpy(),
        "window_start_sec": test.window_start_sec.to_numpy(),
        "true_label": np.where(y == 1, "abnormal", "normal"),
        "probability_abnormal": probabilities,
        "predicted_label": np.where(predictions == 1, "abnormal", "normal"),
    }).to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
