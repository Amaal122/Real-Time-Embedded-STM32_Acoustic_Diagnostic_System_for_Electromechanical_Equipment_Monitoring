#!/usr/bin/env python3
"""Train one recording-level XGBoost model per MIMII slide-rail model ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import entropy
from sklearn.feature_selection import f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def rank_features(X: pd.DataFrame, y: np.ndarray, names: list[str], method: str) -> list[str]:
    if method == "ANOVA":
        raw_scores, _ = f_classif(X[names], y)
        scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        scores = []
        for name in names:
            normal, abnormal = X.loc[y == 0, name].to_numpy(), X.loc[y == 1, name].to_numpy()
            lo, hi = float(min(np.min(normal), np.min(abnormal))), float(max(np.max(normal), np.max(abnormal)))
            if not np.isfinite(lo + hi) or hi - lo < 1e-15:
                scores.append(0.0)
                continue
            edges = np.linspace(lo, hi, 33)
            p = np.histogram(normal, bins=edges)[0].astype(float) + 1e-12
            q = np.histogram(abnormal, bins=edges)[0].astype(float) + 1e-12
            p, q = p / p.sum(), q / q.sum()
            mid = (p + q) / 2
            scores.append(float((entropy(p, mid, base=2) + entropy(q, mid, base=2)) / 2))
        scores = np.asarray(scores)
    return [names[i] for i in np.argsort(-np.asarray(scores), kind="stable")]


def recording_table(rows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    data = pd.DataFrame({"file": rows.file.to_numpy(), "true_label": rows.label.to_numpy(),
                         "probability_abnormal": probabilities})
    return data.groupby("file", sort=False).agg(
        true_label=("true_label", "first"), probability_abnormal=("probability_abnormal", "mean"),
        windows=("true_label", "size"),
    ).reset_index()


def get_threshold(records: pd.DataFrame) -> float:
    y = (records.true_label.to_numpy() == "abnormal").astype(np.int8)
    p = records.probability_abnormal.to_numpy()
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    scores = (tpr + 1.0 - fpr) / 2.0
    return float(thresholds[finite][int(np.argmax(scores[finite]))])


def calculate_metrics(records: pd.DataFrame, threshold: float) -> dict:
    y = (records.true_label.to_numpy() == "abnormal").astype(np.int8)
    p = records.probability_abnormal.to_numpy()
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recordings": len(records), "threshold_from_validation": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_abnormal": float(precision_score(y, pred, zero_division=0)),
        "recall_abnormal": float(recall_score(y, pred, zero_division=0)),
        "f1_abnormal": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_model(y: np.ndarray, seed: int, weighted: bool) -> XGBClassifier:
    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=(neg / pos if weighted else 1.0),
        random_state=seed, n_jobs=-1, tree_method="hist",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--pipeline-metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/per_model_id"))
    parser.add_argument("--model-ids", nargs="+", default=["id_00", "id_02", "id_04", "id_06"])
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--selection-method", choices=["JSD", "ANOVA"], default="JSD")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    available = set(frame.subject.unique())
    if missing := set(args.model_ids) - available:
        raise ValueError(f"Unknown model IDs: {sorted(missing)}")
    metadata_columns = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    feature_names = [c for c in frame.columns if c not in metadata_columns]
    if not 1 <= args.top_k <= len(feature_names):
        raise ValueError(f"--top-k must be between 1 and {len(feature_names)}")
    pipeline = json.loads(args.pipeline_metadata.read_text(encoding="utf-8")) if args.pipeline_metadata.exists() else {}
    summary = []

    for model_id in args.model_ids:
        id_rows = frame[frame.subject == model_id].copy()
        file_labels = id_rows.groupby("file").label.first().reset_index()
        train_files, remaining = train_test_split(
            file_labels, test_size=0.30, random_state=args.seed, stratify=file_labels.label
        )
        val_files, test_files = train_test_split(
            remaining, test_size=0.50, random_state=args.seed, stratify=remaining.label
        )
        train_rows = id_rows[id_rows.file.isin(train_files.file)].copy()
        val_rows = id_rows[id_rows.file.isin(val_files.file)].copy()
        test_rows = id_rows[id_rows.file.isin(test_files.file)].copy()

        def xy(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
            X = rows[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            y = (rows.label.to_numpy() == "abnormal").astype(np.int8)
            return X, y

        X_train, y_train = xy(train_rows)
        X_val, y_val = xy(val_rows)
        X_test, y_test = xy(test_rows)
        ranked_train = rank_features(X_train, y_train, feature_names, args.selection_method)
        selected_train = ranked_train[:args.top_k]

        # Validation chooses class weighting and threshold; both candidate models use train files only.
        candidates = []
        for weighted in (False, True):
            candidate = make_model(y_train, args.seed, weighted)
            candidate.fit(X_train[selected_train], y_train)
            val_prob = candidate.predict_proba(X_val[selected_train])[:, 1]
            val_records = recording_table(val_rows, val_prob)
            threshold = get_threshold(val_records)
            candidates.append((roc_auc_score((val_records.true_label == "abnormal").astype(int),
                                             val_records.probability_abnormal),
                               calculate_metrics(val_records, threshold)["balanced_accuracy"],
                               weighted, threshold))
        _, _, weighted, threshold = max(candidates, key=lambda item: (item[0], item[1]))

        # Re-rank and refit on train+validation; test WAVs remain grouped and held out.
        train_val_rows = pd.concat([train_rows, val_rows], ignore_index=True)
        X_train_val, y_train_val = xy(train_val_rows)
        selected_final = rank_features(X_train_val, y_train_val, feature_names, args.selection_method)[:args.top_k]
        final_model = make_model(y_train_val, args.seed, weighted)
        final_model.fit(X_train_val[selected_final], y_train_val)
        test_prob = final_model.predict_proba(X_test[selected_final])[:, 1]
        test_records = recording_table(test_rows, test_prob)
        test_metrics = calculate_metrics(test_records, threshold)
        test_metrics.update({
            "model_id": model_id,
            "wav_split_counts": {"train": int(len(train_files)), "validation": int(len(val_files)), "test": int(len(test_files))},
            "window_split_counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
            "class_weighting": bool(weighted), "features": len(selected_final),
        })

        model_dir = args.output / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_model.json"
        final_model.save_model(model_path)
        (model_dir / "selected_features.json").write_text(json.dumps(selected_final, indent=2), encoding="utf-8")
        model_metadata = {
            "model_file": model_path.name, "classifier": "XGBoost per MIMII model ID",
            "machine_model_id": model_id, "feature_selection": f"{args.selection_method} fit on train+validation WAV windows",
            "top_k": len(selected_final), "selected_features_file": "selected_features.json",
            "selected_features": selected_final, "label_mapping": {"normal": 0, "abnormal": 1},
            "prediction_threshold": float(threshold),
            "prediction_unit": "recording; mean abnormal probability across windows",
            "fit_files": int(len(train_files) + len(val_files)), "test_files_excluded_from_fit": int(len(test_files)),
            "xgboost_version": xgboost.__version__, "random_seed": args.seed,
            "model_parameters": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                                 "subsample": 0.8, "colsample_bytree": 0.8,
                                 "scale_pos_weight": "validation selected" if weighted else 1.0},
            "preprocessing_from_feature_extraction_run": pipeline,
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
        test_records["predicted_label"] = np.where(test_records.probability_abnormal >= threshold, "abnormal", "normal")
        test_records.to_csv(model_dir / "test_predictions.csv", index=False)
        (model_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        summary.append(test_metrics)
        print(f"{model_id}: {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary_by_model_id.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved one model per MIMII ID to {args.output}", flush=True)


if __name__ == "__main__":
    main()
