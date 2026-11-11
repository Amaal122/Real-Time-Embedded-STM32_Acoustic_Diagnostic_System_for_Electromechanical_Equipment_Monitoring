#!/usr/bin/env python3
"""Evaluate a frozen recording-level model once on a held-out subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier


def calculate_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "recordings": len(y), "threshold_from_validation": threshold,
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_abnormal": float(precision_score(y, predictions, zero_division=0)),
        "recall_abnormal": float(recall_score(y, predictions, zero_division=0)),
        "f1_abnormal": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("xgboost_comparison/recording_level/final_model"))
    parser.add_argument("--test-subject", default="id_06")
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/recording_level/test_evaluation"))
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if args.test_subject in metadata.get("fit_subjects", []):
        raise ValueError("Refusing evaluation: test subject was used to fit this model")
    data = pd.read_csv(args.input)
    test = data[data.subject == args.test_subject].copy()
    del data
    if test.empty:
        raise ValueError(f"No data for test subject {args.test_subject}")
    selected = metadata["selected_features"]
    X = test[selected].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    labels = (test.label.to_numpy() == "abnormal").astype(np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("Test subject must contain both classes")

    model = XGBClassifier()
    model.load_model(args.model_dir / metadata["model_file"])
    window_probabilities = model.predict_proba(X)[:, 1]
    window_rows = pd.DataFrame({
        "subject": test.subject.to_numpy(), "file": test.file.to_numpy(),
        "true_label": test.label.to_numpy(), "probability_abnormal": window_probabilities,
    })
    recordings = window_rows.groupby(["subject", "file"], sort=False).agg(
        probability_abnormal=("probability_abnormal", "mean"),
        true_label=("true_label", "first"), window_count=("true_label", "size"),
    ).reset_index()
    y = (recordings.true_label.to_numpy() == "abnormal").astype(np.int8)
    threshold = float(metadata["prediction_threshold"])
    metrics = calculate_metrics(y, recordings.probability_abnormal.to_numpy(), threshold)
    metrics.update({
        "test_subject": args.test_subject,
        "source_wav_files": int(test.file.nunique()),
        "source_windows": int(len(test)),
        "interpretation_note": "One subject only; the subject appeared in earlier exploratory analyses, so this is retrospective and not a pristine independent test.",
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "recording_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = recordings.copy()
    predictions["predicted_label"] = np.where(predictions.probability_abnormal >= threshold, "abnormal", "normal")
    predictions.to_csv(args.output / "recording_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
