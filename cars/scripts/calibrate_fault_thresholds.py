#!/usr/bin/env python3
"""Calibrate one-vs-rest car-fault thresholds with nested recording-level CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_car_models import METADATA
from train_fault_specialists import FAULT_RULES, normalize_label


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )


def choose_threshold_f1(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], scores)))
    metrics = [(f1_score(y, scores >= threshold, zero_division=0), threshold)
               for threshold in candidates]
    # On an F1 tie choose the larger threshold, which reduces false alarms.
    return float(max(metrics, key=lambda item: (item[0], item[1]))[1])


def choose_threshold_recall(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    candidates = np.unique(np.concatenate(([0.0], scores)))
    viable = []
    for threshold in candidates:
        recall = recall_score(y, scores >= threshold, zero_division=0)
        if recall >= target:
            viable.append(float(threshold))
    return max(viable) if viable else 0.0


def aggregate(rows: pd.DataFrame, window_scores: np.ndarray, targets: dict[str, int]) -> pd.DataFrame:
    temp = rows[["file", "condition", "fault_type", "class_in_state"]].copy().reset_index(drop=True)
    temp["score"] = window_scores
    result = temp.groupby("file", sort=False).agg(
        condition=("condition", "first"), fault_type=("fault_type", "first"),
        class_in_state=("class_in_state", "first"), score=("score", "mean"),
    ).reset_index()
    result["y"] = result.file.map(targets).astype(int)
    return result


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/fault_specialists/threshold_calibration"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.target_recall <= 1:
        parser.error("target-recall must be in (0, 1]")

    data = pd.read_csv(args.input)
    features = [col for col in data.columns if col not in METADATA]
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    recordings = data.groupby("file", sort=True).agg(
        condition=("condition", "first"), fault_type=("fault_type", "first"),
        class_in_state=("class_in_state", "first"), label=("label", "first"),
    ).reset_index()
    normalized = recordings.class_in_state.map(normalize_label)
    output_rows, all_oof = [], []
    threshold_profiles = {}

    for detector_index, (fault, tokens) in enumerate(FAULT_RULES.items()):
        is_positive = normalized.map(lambda value: any(token in value for token in tokens)).to_numpy(dtype=np.int8)
        target_by_file = dict(zip(recordings.file, is_positive))
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_threshold_f1, fold_threshold_recall = [], []

        for fold, (outer_train_idx, outer_test_idx) in enumerate(
            splitter.split(recordings.file, is_positive), start=1
        ):
            outer_train_meta = recordings.iloc[outer_train_idx]
            outer_test_meta = recordings.iloc[outer_test_idx]
            inner_train_meta, calibration_meta = train_test_split(
                outer_train_meta, test_size=0.20, random_state=args.seed + 100 * detector_index + fold,
                stratify=is_positive[outer_train_idx],
            )
            train_files = set(inner_train_meta.file)
            calibration_files = set(calibration_meta.file)
            test_files = set(outer_test_meta.file)
            train_rows = data[data.file.isin(train_files)]
            calibration_rows = data[data.file.isin(calibration_files)]
            test_rows = data[data.file.isin(test_files)]

            y_train = train_rows.file.map(target_by_file).to_numpy(dtype=np.int8)
            model = make_rf(args.seed + detector_index * 100 + fold)
            model.fit(train_rows[features], y_train)
            positive_col = int(np.flatnonzero(model.classes_ == 1)[0])
            calibration = aggregate(
                calibration_rows, model.predict_proba(calibration_rows[features])[:, positive_col], target_by_file
            )
            test = aggregate(test_rows, model.predict_proba(test_rows[features])[:, positive_col], target_by_file)
            val_y, val_scores = calibration.y.to_numpy(), calibration.score.to_numpy()
            test_y, test_scores = test.y.to_numpy(), test.score.to_numpy()
            threshold_f1 = choose_threshold_f1(val_y, val_scores)
            threshold_recall = choose_threshold_recall(val_y, val_scores, args.target_recall)
            fold_threshold_f1.append(threshold_f1)
            fold_threshold_recall.append(threshold_recall)

            for strategy, threshold in (("max_f1", threshold_f1), ("target_recall", threshold_recall)):
                val_metrics = evaluate(val_y, val_scores, threshold)
                test_metrics = evaluate(test_y, test_scores, threshold)
                output_rows.append({
                    "fault": fault, "fold": fold, "strategy": strategy,
                    "calibration_threshold": threshold,
                    "calibration_positive_count": int(val_y.sum()),
                    **{f"calibration_{key}": value for key, value in val_metrics.items()},
                    "test_recordings": int(len(test_y)),
                    "test_positive_count": int(test_y.sum()),
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                })
                test_out = test.copy()
                test_out["fault"] = fault
                test_out["fold"] = fold
                test_out["threshold_strategy"] = strategy
                test_out["threshold"] = threshold
                test_out["prediction"] = (test_out.score >= threshold).astype(int)
                all_oof.append(test_out)
            print(f"{fault} fold {fold}/{args.folds}: max-F1 threshold={threshold_f1:.4f}, "
                  f"80%-recall threshold={threshold_recall:.4f}", flush=True)

        threshold_profiles[fault] = {
            "positive_component_categories": sorted(recordings.loc[is_positive == 1, "fault_type"].unique().tolist()),
            "max_f1_threshold_median": float(np.median(fold_threshold_f1)),
            "target_recall_threshold_median": float(np.median(fold_threshold_recall)),
            "target_recall": args.target_recall,
            "fold_max_f1_thresholds": [float(x) for x in fold_threshold_f1],
            "fold_target_recall_thresholds": [float(x) for x in fold_threshold_recall],
        }

    args.output.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(output_rows)
    fold_frame.to_csv(args.output / "threshold_fold_metrics.csv", index=False)
    oof_frame = pd.concat(all_oof, ignore_index=True)
    oof_frame.to_csv(args.output / "threshold_oof_predictions.csv", index=False)

    pooled_rows = []
    for (fault, strategy), part in oof_frame.groupby(["fault", "threshold_strategy"]):
        truth, scores = part.y.to_numpy(), part.score.to_numpy()
        # Each fold uses its own threshold selected only on the calibration subset.
        pred = part.prediction.to_numpy(dtype=np.int8)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        pooled_rows.append({
            "fault": fault, "strategy": strategy,
            "recordings": int(len(part)), "positive_recordings": int(truth.sum()),
            "roc_auc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "accuracy": float(accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "recall": float(recall_score(truth, pred, zero_division=0)),
            "f1": float(f1_score(truth, pred, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(args.output / "threshold_pooled_oof_metrics.csv", index=False)
    payload = {
        "cv": f"{args.folds}-fold stratified by component presence at WAV level; an inner calibration split chooses each threshold",
        "threshold_strategies": {
            "max_f1": "threshold maximizing F1 on inner calibration recordings",
            "target_recall": f"highest threshold achieving at least {args.target_recall:.0%} recall on inner calibration recordings",
        },
        "threshold_profiles_for_full_data_models": threshold_profiles,
        "note": "Outer-fold predictions use thresholds selected without outer test labels. Median thresholds are starting points for full-data refit models; verify on fresh recordings before deployment.",
    }
    (args.output / "recommended_thresholds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nPooled out-of-fold threshold results:")
    print(pooled[["fault", "strategy", "roc_auc", "average_precision", "balanced_accuracy",
                  "precision", "recall", "f1", "false_positive_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
