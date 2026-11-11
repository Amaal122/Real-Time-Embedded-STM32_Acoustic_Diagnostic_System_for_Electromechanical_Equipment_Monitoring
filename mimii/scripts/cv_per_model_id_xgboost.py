#!/usr/bin/env python3
"""Five-fold file-grouped CV comparing JSD and ANOVA per MIMII model ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def rank_features(X: pd.DataFrame, y: np.ndarray, names: list[str], method: str) -> list[str]:
    if method == "ANOVA":
        values, _ = f_classif(X[names], y)
        scores = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        scores = []
        for name in names:
            normal = X.loc[y == 0, name].to_numpy()
            abnormal = X.loc[y == 1, name].to_numpy()
            low, high = float(min(normal.min(), abnormal.min())), float(max(normal.max(), abnormal.max()))
            if not np.isfinite(low + high) or high - low < 1e-15:
                scores.append(0.0)
                continue
            bins = np.linspace(low, high, 33)
            p = np.histogram(normal, bins=bins)[0].astype(float) + 1e-12
            q = np.histogram(abnormal, bins=bins)[0].astype(float) + 1e-12
            p, q = p / p.sum(), q / q.sum()
            mid = (p + q) / 2
            scores.append(float((entropy(p, mid, base=2) + entropy(q, mid, base=2)) / 2))
        scores = np.asarray(scores)
    return [names[i] for i in np.argsort(-np.asarray(scores), kind="stable")]


def aggregate(files: pd.Series, labels: np.ndarray, probabilities: np.ndarray, fold: int,
              model_id: str, method: str) -> pd.DataFrame:
    rows = pd.DataFrame({"file": files.to_numpy(), "true_label": np.where(labels == 1, "abnormal", "normal"),
                         "probability_abnormal": probabilities})
    result = rows.groupby("file", sort=False).agg(
        true_label=("true_label", "first"), probability_abnormal=("probability_abnormal", "mean"),
        windows=("true_label", "size"),
    ).reset_index()
    result["fold"] = fold
    result["model_id"] = model_id
    result["method"] = method
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(np.int8)
    return {
        "recordings": len(y), "roc_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
        "precision_abnormal_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_abnormal_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "f1_abnormal_at_0_5": float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/per_model_id_cv"))
    parser.add_argument("--model-ids", nargs="+", default=["id_00", "id_02", "id_04", "id_06"])
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    meta = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    features = [c for c in data.columns if c not in meta]
    all_oof, fold_metrics = [], []
    for model_id in args.model_ids:
        id_data = data[data.subject == model_id].copy()
        file_labels = id_data.groupby("file", sort=True).label.first().reset_index()
        if file_labels.empty or file_labels.label.value_counts().min() < args.folds:
            raise ValueError(f"Not enough files per class for {args.folds}-fold CV on {model_id}")
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(file_labels.file, file_labels.label), start=1):
            train_files = set(file_labels.iloc[train_idx].file)
            val_files = set(file_labels.iloc[val_idx].file)
            train_rows = id_data[id_data.file.isin(train_files)]
            val_rows = id_data[id_data.file.isin(val_files)]
            X_train = train_rows[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            y_train = (train_rows.label.to_numpy() == "abnormal").astype(np.int8)
            X_val = val_rows[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            y_val = (val_rows.label.to_numpy() == "abnormal").astype(np.int8)
            for method in ("JSD", "ANOVA"):
                selected = rank_features(X_train, y_train, features, method)[:args.top_k]
                model = XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss", n_estimators=200,
                    max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, scale_pos_weight=1.0,
                    random_state=args.seed + fold, n_jobs=-1, tree_method="hist",
                )
                model.fit(X_train[selected], y_train)
                probabilities = model.predict_proba(X_val[selected])[:, 1]
                oof = aggregate(val_rows.file, y_val, probabilities, fold, model_id, method)
                all_oof.append(oof)
                score = metrics((oof.true_label == "abnormal").astype(np.int8).to_numpy(),
                                oof.probability_abnormal.to_numpy())
                fold_metrics.append({"model_id": model_id, "method": method, "fold": fold, **score})
            print(f"Finished {model_id} fold {fold}/{args.folds}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    oof_data = pd.concat(all_oof, ignore_index=True)
    fold_data = pd.DataFrame(fold_metrics)
    oof_data.to_csv(args.output / "recording_oof_predictions.csv", index=False)
    fold_data.to_csv(args.output / "fold_metrics.csv", index=False)
    summary = []
    for (model_id, method), rows in oof_data.groupby(["model_id", "method"]):
        y = (rows.true_label == "abnormal").astype(np.int8).to_numpy()
        p = rows.probability_abnormal.to_numpy()
        summary.append({"model_id": model_id, "method": method, **metrics(y, p)})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output / "summary_by_model_id.csv", index=False)
    aggregate_summary = summary_frame.groupby("method").agg(
        macro_roc_auc=("roc_auc", "mean"), macro_balanced_accuracy=("balanced_accuracy_at_0_5", "mean"),
        macro_precision=("precision_abnormal_at_0_5", "mean"), macro_recall=("recall_abnormal_at_0_5", "mean"),
    ).reset_index()
    aggregate_summary.to_csv(args.output / "macro_summary.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "evaluation": "5-fold stratified CV grouped by WAV, run separately within each MIMII model ID",
        "features_per_method": args.top_k, "methods": ["JSD", "ANOVA"], "folds": args.folds,
        "xgboost_parameters": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                               "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": 1.0},
        "previous_exploration_used_all_recordings": True,
    }, indent=2), encoding="utf-8")
    print(aggregate_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
