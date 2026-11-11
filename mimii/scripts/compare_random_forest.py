#!/usr/bin/env python3
"""Evaluate Random Forest on the 0 dB slider features with subject-held-out CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut

from compare_xgboost_selection import rank_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("random_forest_comparison"))
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metadata = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    names = [col for col in frame.columns if col not in metadata]
    if len(names) != 65:
        raise ValueError(f"Expected 65 feature columns, found {len(names)}")
    groups = frame.subject.to_numpy()
    subjects = sorted(frame.subject.unique())
    X = frame[names].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = (frame.label.to_numpy() == "abnormal").astype(np.int8)
    splitter = LeaveOneGroupOut()
    results: list[dict] = []
    selected_by_fold: dict[str, list[str]] = {}
    folds = list(splitter.split(X, y, groups))

    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        train_y = y[train_idx]
        train_X = X.iloc[train_idx]
        rankings = rank_features(train_X, train_y, names)
        cases = [(method, k, rankings[method][:k])
                 for method in ("JSD", "ANOVA") for k in sorted(set(args.top_k))]
        cases.append(("ALL", len(names), names))
        for method, k, selected in cases:
            model = RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", max_features="sqrt",
                min_samples_leaf=1, n_jobs=-1, random_state=args.seed,
            )
            model.fit(train_X[selected], train_y)
            probabilities = model.predict_proba(X.iloc[test_idx][selected])[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int8)
            tn, fp, fn, tp = confusion_matrix(y[test_idx], predictions, labels=[0, 1]).ravel()
            results.append({
                "fold": fold_index, "test_subject": str(groups[test_idx[0]]), "method": method,
                "top_k": k, "n_train": len(train_idx), "n_test": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], predictions),
                "balanced_accuracy": balanced_accuracy_score(y[test_idx], predictions),
                "precision_abnormal": precision_score(y[test_idx], predictions, zero_division=0),
                "recall_abnormal": recall_score(y[test_idx], predictions, zero_division=0),
                "f1_abnormal": f1_score(y[test_idx], predictions, zero_division=0),
                "roc_auc": roc_auc_score(y[test_idx], probabilities),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
            selected_by_fold[f"fold_{fold_index}/{method}/top_{k}"] = selected
        print(f"Finished fold {fold_index}/{len(folds)}; held-out subject: {groups[test_idx[0]]}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(args.output / "fold_results.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "precision_abnormal", "recall_abnormal", "f1_abnormal", "roc_auc"]
    summary = results_frame.groupby(["method", "top_k"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    (args.output / "selected_features_by_fold.json").write_text(json.dumps(selected_by_fold, indent=2), encoding="utf-8")
    (args.output / "evaluation_metadata.json").write_text(json.dumps({
        "classifier": "RandomForestClassifier", "evaluation": "LeaveOneGroupOut by subject",
        "subjects": subjects, "features": len(names), "top_k_values": sorted(set(args.top_k)),
        "positive_class": "abnormal", "threshold": 0.5,
        "feature_selection_refit_inside_each_training_fold": True,
        "model_parameters": {"n_estimators": 300, "class_weight": "balanced_subsample",
                             "max_features": "sqrt", "min_samples_leaf": 1},
        "random_seed": args.seed,
        "comparison_note": "Exploratory; uses the same retrospective dataset and folds as the XGBoost comparison.",
    }, indent=2), encoding="utf-8")
    print(f"Saved fold results and summary to {args.output}", flush=True)


if __name__ == "__main__":
    main()
