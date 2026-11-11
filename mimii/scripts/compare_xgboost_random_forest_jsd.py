#!/usr/bin/env python3
"""Compare XGBoost and Random Forest with JSD top-k on a grouped mixed-ID split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


def jsd_score(normal: np.ndarray, abnormal: np.ndarray, bins: int = 32) -> float:
    low = float(min(normal.min(), abnormal.min()))
    high = float(max(normal.max(), abnormal.max()))
    if not np.isfinite(low + high) or high - low < 1e-15:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    p = np.histogram(normal, bins=edges)[0].astype(float) + 1e-12
    q = np.histogram(abnormal, bins=edges)[0].astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    middle = (p + q) / 2
    return float((entropy(p, middle, base=2) + entropy(q, middle, base=2)) / 2)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predicted = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_abnormal": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_abnormal": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_abnormal": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("random_forest_comparison/mixed_ids_grouped_split"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metadata = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    features = [column for column in frame.columns if column not in metadata]
    if set(frame.label.unique()) != {"normal", "abnormal"}:
        raise ValueError("Input must contain normal and abnormal labels")
    X = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = (frame.label.to_numpy() == "abnormal").astype(np.int8)
    # A recording's overlapping windows must remain together. Subject is included
    # in the key so identically named files from different IDs cannot collide.
    groups = frame.subject.astype(str) + "/" + frame.file.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    train_y = y[train_idx]
    normal = train_y == 0
    abnormal = train_y == 1
    scores = [jsd_score(X.iloc[train_idx][name].to_numpy()[normal],
                        X.iloc[train_idx][name].to_numpy()[abnormal]) for name in features]
    selected = [features[i] for i in np.argsort(-np.asarray(scores), kind="stable")[:args.top_k]]

    models = {
        "XGBoost_JSD": XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=200,
            max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=args.seed, n_jobs=-1, tree_method="hist",
        ),
        "RandomForest_JSD": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1, random_state=args.seed,
        ),
    }
    row_results = []
    recording_results = []
    for name, model in models.items():
        model.fit(X.iloc[train_idx][selected], train_y)
        probabilities = model.predict_proba(X.iloc[test_idx][selected])[:, 1]
        row_results.append({"model": name, **metrics(y[test_idx], probabilities)})
        test_frame = frame.iloc[test_idx][["subject", "file"]].copy()
        test_frame["label"] = y[test_idx]
        test_frame["probability_abnormal"] = probabilities
        recordings = test_frame.groupby(["subject", "file"], as_index=False).agg(
            label=("label", "first"), probability_abnormal=("probability_abnormal", "mean"))
        recording_results.append({"model": name, **metrics(
            recordings.label.to_numpy(), recordings.probability_abnormal.to_numpy())})

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(row_results).to_csv(args.output / "window_metrics.csv", index=False)
    pd.DataFrame(recording_results).to_csv(args.output / "recording_metrics.csv", index=False)
    (args.output / "split_metadata.json").write_text(json.dumps({
        "evaluation": "single mixed-ID GroupShuffleSplit grouped by WAV recording",
        "test_size_requested": args.test_size, "seed": args.seed, "top_k": args.top_k,
        "feature_selection": "JSD fit on training rows only",
        "ids": sorted(frame.subject.astype(str).unique().tolist()),
        "train_ids": sorted(frame.iloc[train_idx].subject.astype(str).unique().tolist()),
        "test_ids": sorted(frame.iloc[test_idx].subject.astype(str).unique().tolist()),
        "train_recordings": int(groups.iloc[train_idx].nunique()),
        "test_recordings": int(groups.iloc[test_idx].nunique()),
        "train_windows": int(len(train_idx)), "test_windows": int(len(test_idx)),
        "selected_features": selected,
        "threshold": 0.5,
        "models": {"XGBoost_JSD": {"n_estimators": 200, "max_depth": 4,
                     "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
                   "RandomForest_JSD": {"n_estimators": 300,
                     "class_weight": "balanced_subsample", "max_features": "sqrt"}},
        "note": "One exploratory split; report recording metrics as primary because windows overlap.",
    }, indent=2), encoding="utf-8")
    print("Window metrics:")
    print(pd.DataFrame(row_results).to_string(index=False))
    print("\nRecording-level metrics:")
    print(pd.DataFrame(recording_results).to_string(index=False))
    print(f"\nSaved outputs to {args.output}")


if __name__ == "__main__":
    main()
