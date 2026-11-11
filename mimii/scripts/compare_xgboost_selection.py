#!/usr/bin/env python3
"""Compare JSD and ANOVA feature rankings with subject-held-out XGBoost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier


def jsd_score(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo + hi) or hi - lo < 1e-15:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
    pb = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    mid = (pa + pb) / 2
    return float((entropy(pa, mid, base=2) + entropy(pb, mid, base=2)) / 2)


def rank_features(X: pd.DataFrame, y: np.ndarray, names: list[str]) -> dict[str, list[str]]:
    # Labels are 0=normal and 1=abnormal; fit rankings on this fold's training data only.
    normal = y == 0
    abnormal = y == 1
    jsd = [jsd_score(X.loc[normal, name].to_numpy(), X.loc[abnormal, name].to_numpy()) for name in names]
    f_values, _ = f_classif(X[names], y)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "JSD": [names[i] for i in np.argsort(-np.asarray(jsd), kind="stable")],
        "ANOVA": [names[i] for i in np.argsort(-f_values, kind="stable")],
    }


def make_model(y_train: np.ndarray, seed: int) -> XGBClassifier:
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=negatives / positives,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison"))
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metadata = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    names = [col for col in frame.columns if col not in metadata]
    if len(names) != 65:
        raise ValueError(f"Expected 65 feature columns, found {len(names)}")
    if set(frame.label.unique()) != {"normal", "abnormal"}:
        raise ValueError("Input must contain both normal and abnormal classes")
    groups = frame.subject.to_numpy()
    subjects = sorted(frame.subject.unique())
    if len(subjects) < 2:
        raise ValueError("At least two subjects are required for subject-held-out evaluation")
    if any(k < 1 or k > len(names) for k in args.top_k):
        raise ValueError(f"--top-k values must be between 1 and {len(names)}")

    X = frame[names].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = (frame.label.to_numpy() == "abnormal").astype(np.int8)
    splitter = LeaveOneGroupOut()
    results: list[dict] = []
    selected_by_fold: dict[str, dict[str, list[str]]] = {}
    folds = list(splitter.split(X, y, groups))

    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        train_y, test_y = y[train_idx], y[test_idx]
        test_subject = str(groups[test_idx[0]])
        train_X = X.iloc[train_idx]
        rankings = rank_features(train_X, train_y, names)
        cases: list[tuple[str, int | None, list[str]]]
        cases = [(method, k, rankings[method][:k]) for method in ("JSD", "ANOVA") for k in sorted(set(args.top_k))]
        cases.append(("ALL", len(names), names))

        for method, k, selected in cases:
            model = make_model(train_y, args.seed)
            model.fit(train_X[selected], train_y)
            probabilities = model.predict_proba(X.iloc[test_idx][selected])[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int8)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions, labels=[0, 1]).ravel()
            results.append({
                "fold": fold_index, "test_subject": test_subject, "method": method,
                "top_k": k, "n_train": len(train_idx), "n_test": len(test_idx),
                "accuracy": accuracy_score(test_y, predictions),
                "balanced_accuracy": balanced_accuracy_score(test_y, predictions),
                "precision_abnormal": precision_score(test_y, predictions, zero_division=0),
                "recall_abnormal": recall_score(test_y, predictions, zero_division=0),
                "f1_abnormal": f1_score(test_y, predictions, zero_division=0),
                "roc_auc": roc_auc_score(test_y, probabilities),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
            selected_by_fold[f"fold_{fold_index}/{method}/top_{k}"] = selected
        print(f"Finished fold {fold_index}/{len(folds)}; held-out subject: {test_subject}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(args.output / "fold_results.csv", index=False)
    metric_names = ["accuracy", "balanced_accuracy", "precision_abnormal", "recall_abnormal", "f1_abnormal", "roc_auc"]
    summary = results_frame.groupby(["method", "top_k"])[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    (args.output / "selected_features_by_fold.json").write_text(json.dumps(selected_by_fold, indent=2), encoding="utf-8")
    (args.output / "evaluation_metadata.json").write_text(json.dumps({
        "classifier": "XGBoost", "evaluation": "LeaveOneGroupOut by subject",
        "subjects": subjects, "features": len(names), "top_k_values": sorted(set(args.top_k)),
        "positive_class": "abnormal", "threshold": 0.5,
        "feature_selection_refit_inside_each_training_fold": True,
        "model_parameters": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                             "subsample": 0.8, "colsample_bytree": 0.8,
                             "scale_pos_weight": "negative/positive training rows"},
        "random_seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"Saved fold results and summary to {args.output}", flush=True)


if __name__ == "__main__":
    main()
