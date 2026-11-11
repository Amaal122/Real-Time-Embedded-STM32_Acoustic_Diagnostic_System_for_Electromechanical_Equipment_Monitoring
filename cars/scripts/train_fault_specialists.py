#!/usr/bin/env python3
"""Train one binary RandomForest detector per car fault component."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
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
from sklearn.model_selection import StratifiedKFold

from train_car_models import METADATA


FAULT_RULES = {
    "worn_out_brakes": ["worn out brakes"],
    "bad_ignition": ["bad ignition"],
    "dead_battery": ["dead battery"],
    "low_oil": ["low oil", "no oil"],
    "power_steering": ["power steering"],
    "serpentine_belt": ["serpentine belt"],
}


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recordings": int(len(y)),
        "positive_recordings": int(np.sum(y == 1)),
        "negative_recordings": int(np.sum(y == 0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def aggregate_file_scores(rows: pd.DataFrame, window_scores: np.ndarray,
                          target_by_file: dict[str, int]) -> pd.DataFrame:
    temp = rows[["file", "condition", "fault_type", "class_in_state"]].copy().reset_index(drop=True)
    temp["score"] = window_scores
    result = temp.groupby("file", sort=False).agg(
        condition=("condition", "first"), fault_type=("fault_type", "first"),
        class_in_state=("class_in_state", "first"), score=("score", "mean"),
    ).reset_index()
    result["target"] = result.file.map(target_by_file).astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/fault_specialists"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fixed decision threshold used for CV metrics and saved detector metadata")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    features = [col for col in data.columns if col not in METADATA]
    if len(features) != 65:
        raise ValueError(f"Expected 65 audio features, found {len(features)}")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    recordings = data.groupby("file", sort=True).agg(
        condition=("condition", "first"), fault_type=("fault_type", "first"),
        class_in_state=("class_in_state", "first"), label=("label", "first"),
    ).reset_index()
    normalized = recordings.class_in_state.map(normalize_label)

    targets: dict[str, dict[str, int]] = {}
    positive_categories: dict[str, list[str]] = {}
    for fault, tokens in FAULT_RULES.items():
        positives = normalized.map(lambda value: any(token in value for token in tokens))
        targets[fault] = dict(zip(recordings.file, positives.astype(int)))
        positive_categories[fault] = sorted(recordings.loc[positives, "fault_type"].unique().tolist())

    args.output.mkdir(parents=True, exist_ok=True)
    fold_rows, pooled_rows, model_files = [], [], {}
    for fault, file_targets in targets.items():
        y_recordings = recordings.file.map(file_targets).to_numpy(dtype=np.int8)
        if min(np.bincount(y_recordings, minlength=2)) < args.folds:
            raise ValueError(f"Not enough recordings in both classes for {fault} {args.folds}-fold CV")
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        all_scores = np.full(len(recordings), np.nan, dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(recordings.file, y_recordings), start=1):
            train_files = set(recordings.iloc[train_idx].file)
            val_files = set(recordings.iloc[val_idx].file)
            train_rows = data[data.file.isin(train_files)]
            val_rows = data[data.file.isin(val_files)]
            y_train = train_rows.file.map(file_targets).to_numpy(dtype=np.int8)
            model = make_rf(args.seed + fold)
            model.fit(train_rows[features], y_train)
            positive_col = int(np.flatnonzero(model.classes_ == 1)[0])
            window_scores = model.predict_proba(val_rows[features])[:, positive_col]
            val_agg = aggregate_file_scores(val_rows, window_scores, file_targets)
            all_scores[val_idx] = val_agg.set_index("file").loc[recordings.iloc[val_idx].file, "score"].to_numpy()
            fold_metric = metrics(y_recordings[val_idx], all_scores[val_idx], args.threshold)
            fold_metric.update({"fault": fault, "fold": fold})
            fold_rows.append(fold_metric)

        if np.isnan(all_scores).any():
            raise AssertionError(f"Some recordings did not receive an out-of-fold score for {fault}")
        pooled_metric = metrics(y_recordings, all_scores, args.threshold)
        pooled_metric.update({"fault": fault, "positive_categories": positive_categories[fault]})
        pooled_rows.append(pooled_metric)

        prediction = recordings[["file", "condition", "fault_type", "class_in_state"]].copy()
        prediction["true_fault"] = y_recordings
        prediction["oof_probability"] = all_scores
        prediction["predicted_fault"] = (all_scores >= args.threshold).astype(np.int8)
        prediction.to_csv(args.output / f"oof_predictions_{fault}.csv", index=False)

        # Fit a deployable specialist on every labelled recording after CV evaluation.
        final_model = make_rf(args.seed)
        y_all_rows = data.file.map(file_targets).to_numpy(dtype=np.int8)
        final_model.fit(data[features], y_all_rows)
        bundle = {
            "model": final_model,
            "fault": fault,
            "features": features,
            "positive_folder_categories": positive_categories[fault],
            "threshold": float(args.threshold),
            "positive_mapping": "one if any listed fault component is present; combined recordings count positive for every component they contain",
            "seed": args.seed,
        }
        path = args.output / f"detector_{fault}.joblib"
        joblib.dump(bundle, path)
        model_files[fault] = path.name
        print(f"Finished {fault}: positive WAVs={int(y_recordings.sum())}, "
              f"OOF F1={pooled_metric['f1']:.3f}, AP={pooled_metric['average_precision']:.3f}", flush=True)

    fold_frame = pd.DataFrame(fold_rows)
    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame.to_csv(args.output / "fold_metrics.csv", index=False)
    pooled_frame.to_csv(args.output / "specialist_summary.csv", index=False)
    mapping = {
        "recordings": int(len(recordings)),
        "windows": int(len(data)),
        "folds": args.folds,
        "threshold": args.threshold,
        "split_unit": "original WAV file; windows from a WAV stay in one fold",
        "classifier": "RandomForestClassifier, 400 trees, class_weight=balanced, all 65 features",
        "fault_mapping": {fault: {"positive_folders": positive_categories[fault], "model_file": model_files[fault]}
                          for fault in FAULT_RULES},
        "combined_fault_policy": "a combination WAV is positive for each component named in its folder",
        "evaluation_note": "Pooled out-of-fold scores are cross-validated estimates; final saved models are refit on all labelled WAVs.",
    }
    (args.output / "mapping_and_run_metadata.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print("\nFive-fold specialist summary:")
    print(pooled_frame[["fault", "positive_recordings", "negative_recordings", "roc_auc",
                        "average_precision", "balanced_accuracy", "precision", "recall", "f1"]].to_string(index=False))
    print(f"Saved specialist models and evaluation files to {args.output.resolve()}")


if __name__ == "__main__":
    main()
