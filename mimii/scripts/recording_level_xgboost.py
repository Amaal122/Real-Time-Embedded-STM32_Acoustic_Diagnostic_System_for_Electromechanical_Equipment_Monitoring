#!/usr/bin/env python3
"""Fit a window model but evaluate/decide at recording level using train/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import entropy
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier


def jsd_rank(X: pd.DataFrame, y: np.ndarray, features: list[str]) -> list[str]:
    scores = []
    for feature in features:
        a, b = X.loc[y == 0, feature].to_numpy(), X.loc[y == 1, feature].to_numpy()
        lo, hi = float(min(np.min(a), np.min(b))), float(max(np.max(a), np.max(b)))
        if not np.isfinite(lo + hi) or hi - lo < 1e-15:
            scores.append(0.0)
            continue
        edges = np.linspace(lo, hi, 33)
        pa = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
        pb = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
        pa /= pa.sum()
        pb /= pb.sum()
        mid = (pa + pb) / 2
        scores.append(float((entropy(pa, mid, base=2) + entropy(pb, mid, base=2)) / 2))
    return [features[i] for i in np.argsort(-np.asarray(scores), kind="stable")]


def recording_predictions(part: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    windows = pd.DataFrame({
        "file": part.file.to_numpy(), "subject": part.subject.to_numpy(),
        "true_label": part.label.to_numpy(), "probability_abnormal": probabilities,
    })
    grouped = windows.groupby(["subject", "file"], sort=False).agg(
        probability_abnormal=("probability_abnormal", "mean"),
        true_label=("true_label", "first"), window_count=("true_label", "size"),
    ).reset_index()
    return grouped


def score_records(records: pd.DataFrame, threshold: float) -> dict:
    y = (records.true_label.to_numpy() == "abnormal").astype(np.int8)
    p = records.probability_abnormal.to_numpy()
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recordings": len(records), "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_abnormal": float(precision_score(y, pred, zero_division=0)),
        "recall_abnormal": float(recall_score(y, pred, zero_division=0)),
        "f1_abnormal": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--pipeline-metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/recording_level"))
    parser.add_argument("--train-subjects", nargs="+", default=["id_00", "id_02"])
    parser.add_argument("--validation-subject", default="id_04")
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Explicitly restrict model selection/evaluation data to train and validation subjects.
    all_rows = pd.read_csv(args.input)
    allowed = set(args.train_subjects) | {args.validation_subject}
    frame = all_rows[all_rows.subject.isin(allowed)].copy()
    del all_rows
    train = frame[frame.subject.isin(args.train_subjects)].copy()
    val = frame[frame.subject == args.validation_subject].copy()
    if train.empty or val.empty:
        raise ValueError("Training and validation subjects must exist in the input")
    excluded = set(frame.subject.unique()) - allowed
    if excluded:
        raise ValueError("Unexpected subject in model-selection rows")
    metadata_columns = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    features = [c for c in frame.columns if c not in metadata_columns]
    if not 1 <= args.top_k <= len(features):
        raise ValueError(f"--top-k must be between 1 and {len(features)}")

    def xy(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return (part[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32),
                (part.label.to_numpy() == "abnormal").astype(np.int8))

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    ranked_train = jsd_rank(X_train, y_train, features)
    selected_train = ranked_train[:args.top_k]
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 200,
              "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "reg_lambda": 1.0, "scale_pos_weight": 1.0,
              "random_state": args.seed, "n_jobs": -1, "tree_method": "hist"}
    validation_model = XGBClassifier(**params)
    validation_model.fit(X_train[selected_train], y_train)
    val_probs = validation_model.predict_proba(X_val[selected_train])[:, 1]
    val_records = recording_predictions(val, val_probs)
    val_y = (val_records.true_label.to_numpy() == "abnormal").astype(np.int8)
    fpr, tpr, thresholds = roc_curve(val_y, val_records.probability_abnormal.to_numpy())
    finite = np.isfinite(thresholds)
    balanced_scores = (tpr + 1 - fpr) / 2
    threshold = float(thresholds[finite][int(np.argmax(balanced_scores[finite]))])
    val_metrics_tuned = score_records(val_records, threshold)
    val_metrics_05 = score_records(val_records, 0.5)

    # Final recording classifier fit on train+validation; the test subject is excluded.
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val, y_train_val = xy(train_val)
    selected_final = jsd_rank(X_train_val, y_train_val, features)[:args.top_k]
    final_model = XGBClassifier(**params)
    final_model.fit(X_train_val[selected_final], y_train_val)

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_model.json"
    final_model.save_model(model_path)
    (model_dir / "selected_features.json").write_text(json.dumps(selected_final, indent=2), encoding="utf-8")
    pipeline = json.loads(args.pipeline_metadata.read_text(encoding="utf-8")) if args.pipeline_metadata.exists() else {}
    model_metadata = {
        "model_file": model_path.name, "classifier": "XGBoost",
        "feature_selection": "JSD fit on train+validation windows", "top_k": args.top_k,
        "selected_features_file": "selected_features.json", "selected_features": selected_final,
        "prediction_unit": "recording; mean abnormal probability across windows",
        "label_mapping": {"normal": 0, "abnormal": 1}, "prediction_threshold": threshold,
        "threshold_tuned_on": f"recording-level validation predictions for {args.validation_subject}",
        "fit_subjects": sorted(set(args.train_subjects) | {args.validation_subject}),
        "xgboost_version": xgboost.__version__, "random_seed": args.seed,
        "model_parameters": params, "preprocessing_from_feature_extraction_run": pipeline,
        "note": "Selected and thresholded using training/validation subjects only. Test subject was excluded.",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    val_records["predicted_label"] = np.where(val_records.probability_abnormal >= threshold, "abnormal", "normal")
    val_records.to_csv(args.output / "validation_recording_predictions.csv", index=False)
    (args.output / "validation_metrics.json").write_text(json.dumps({
        "mean_probability_threshold": val_metrics_tuned,
        "fixed_0_5_threshold": val_metrics_05,
        "n_subjects_train": len(set(args.train_subjects)), "validation_subject": args.validation_subject,
    }, indent=2), encoding="utf-8")
    print(f"Train recordings: {train.file.nunique()}, validation recordings: {val.file.nunique()}")
    print("Validation at file level: " + json.dumps(val_metrics_tuned, sort_keys=True), flush=True)
    print(f"Saved recording-level model to {model_dir}; test subject was not used.", flush=True)


if __name__ == "__main__":
    main()
