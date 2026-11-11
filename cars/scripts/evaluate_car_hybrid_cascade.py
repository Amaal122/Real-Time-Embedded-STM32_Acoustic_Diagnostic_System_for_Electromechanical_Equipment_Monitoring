#!/usr/bin/env python3
"""Evaluate XGBoost normal/anomaly detection followed by RandomForest fault diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train_car_models import METADATA


def rank_anova(x: pd.DataFrame, y: np.ndarray, feature_names: list[str], top_k: int) -> list[str]:
    scores, _ = f_classif(x[feature_names], y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-scores, kind="stable")[:min(top_k, len(feature_names))]
    return [feature_names[index] for index in order]


def make_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=300,
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, tree_method="hist",
    )


def fit_detector(train_rows: pd.DataFrame, val_rows: pd.DataFrame, feature_names: list[str],
                 top_k: int, seed: int) -> tuple[XGBClassifier, list[str], float]:
    y = (train_rows.label == "abnormal").astype(int).to_numpy()
    selected = rank_anova(train_rows[feature_names], y, feature_names, top_k)
    model = make_xgb(seed)
    model.fit(train_rows[selected], y, sample_weight=compute_sample_weight("balanced", y))
    val = aggregate_detector(val_rows, model.predict_proba(val_rows[selected])[:, 1])
    normal_scores = val.loc[val.true_binary == 0, "anomaly_score"].to_numpy()
    threshold = float(np.quantile(normal_scores, 0.95, method="higher"))
    return model, selected, threshold


def aggregate_detector(rows: pd.DataFrame, window_scores: np.ndarray) -> pd.DataFrame:
    temp = rows[["file", "label", "fault_type", "condition"]].copy().reset_index(drop=True)
    temp["true_binary"] = (temp.label == "abnormal").astype(int)
    temp["anomaly_score"] = window_scores
    return temp.groupby("file", sort=False).agg(
        label=("label", "first"), fault_type=("fault_type", "first"),
        condition=("condition", "first"), true_binary=("true_binary", "first"),
        anomaly_score=("anomaly_score", "mean"),
    ).reset_index()


def fit_fault_model(train_rows: pd.DataFrame, feature_names: list[str], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    abnormal = train_rows[train_rows.label == "abnormal"]
    classes = sorted(abnormal.fault_type.unique().tolist())
    model = RandomForestClassifier(
        n_estimators=400, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    model.fit(abnormal[feature_names], abnormal.fault_type)
    return model, classes


def aggregate_fault(rows: pd.DataFrame, model, feature_names: list[str], class_names: list[str]) -> pd.DataFrame:
    scores = model.predict_proba(rows[feature_names])
    cols = [f"p_{i}" for i in range(len(class_names))]
    temp = rows[["file", "label", "fault_type", "condition"]].copy().reset_index(drop=True)
    temp = pd.concat([temp, pd.DataFrame(scores, columns=cols)], axis=1)
    result = temp.groupby("file", sort=False).agg(
        label=("label", "first"), fault_type=("fault_type", "first"),
        condition=("condition", "first"), **{col: (col, "mean") for col in cols},
    ).reset_index()
    result["fault_prediction"] = np.asarray(class_names, dtype=object)[
        np.argmax(result[cols].to_numpy(), axis=1)
    ]
    return result


def form_cascade(detector_rows: pd.DataFrame, fault_rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    detector_rows = detector_rows.copy()
    detector_rows["stage1_prediction"] = np.where(
        detector_rows.anomaly_score >= threshold, "abnormal", "normal"
    )
    result = detector_rows[["file", "label", "fault_type", "condition", "true_binary",
                            "anomaly_score", "stage1_prediction"]].merge(
        fault_rows[["file", "fault_prediction"]], on="file", validate="one_to_one"
    )
    result["true_output"] = np.where(result.label == "normal", "normal", result.fault_type)
    result["predicted_output"] = np.where(
        result.stage1_prediction == "normal", "normal", result.fault_prediction
    )
    return result


def summarize_cascade(rows: pd.DataFrame, class_names: list[str]) -> dict:
    truth = rows.true_output.to_numpy()
    pred = rows.predicted_output.to_numpy()
    return {
        "recordings": int(len(rows)),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, labels=class_names,
                                                  average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=class_names,
                                            average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, labels=class_names,
                                    average="macro", zero_division=0)),
    }


def evaluate_saved_holdout(data: pd.DataFrame, features: list[str], split_file: Path,
                           output: Path, model_root: Path) -> dict:
    split = pd.read_csv(split_file)
    test_files = set(split.loc[split.split == "test", "file"])
    test_rows = data[data.file.isin(test_files)].copy()
    detector_meta_path = model_root / "xgboost_binary_anova_metadata.json"
    detector_meta = json.loads(detector_meta_path.read_text(encoding="utf-8"))
    selected = detector_meta["selected_features"]
    detector = XGBClassifier()
    detector.load_model(model_root / "xgboost_binary_anova.json")
    detector_rows = aggregate_detector(test_rows, detector.predict_proba(test_rows[selected])[:, 1])

    saved = joblib.load(model_root / "fault_model_search" / "selected_fault_classifier.joblib")
    fault_rows = aggregate_fault(test_rows, saved["model"], saved["features"], saved["classes"])
    cascade = form_cascade(detector_rows, fault_rows, float(detector_meta["threshold"]))
    class_names = ["normal"] + saved["classes"]
    metrics = summarize_cascade(cascade, class_names)
    output.mkdir(parents=True, exist_ok=True)
    cascade.to_csv(output / "holdout_test_predictions.csv", index=False)
    cm = confusion_matrix(cascade.true_output, cascade.predicted_output, labels=class_names)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output / "holdout_confusion_matrix.csv")
    result = {"detector": "XGBoost ANOVA top-60", "fault_classifier": "Random Forest, all 65 features",
              "threshold": float(detector_meta["threshold"]), "classes": class_names, "metrics": metrics,
              "note": "Retrospective held-out split; previous models were also compared on this split."}
    (output / "holdout_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_nested_grouped_cv(data: pd.DataFrame, feature_names: list[str], output: Path,
                               n_splits: int, top_k: int, seed: int) -> dict:
    recordings = data.groupby("file", sort=True).agg(
        fault_type=("fault_type", "first"), label=("label", "first"), condition=("condition", "first")
    ).reset_index()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    class_names = ["normal"] + sorted(recordings.loc[recordings.label == "abnormal", "fault_type"].unique())
    fold_summaries, all_predictions = [], []

    for fold, (outer_train_indices, outer_test_indices) in enumerate(
        splitter.split(recordings.file, recordings.fault_type), start=1
    ):
        outer_train_meta = recordings.iloc[outer_train_indices].copy()
        outer_test_meta = recordings.iloc[outer_test_indices].copy()
        inner_train_meta, calibration_meta = train_test_split(
            outer_train_meta, test_size=0.20, random_state=seed + fold,
            stratify=outer_train_meta.fault_type,
        )
        train_files = set(inner_train_meta.file)
        calibration_files = set(calibration_meta.file)
        test_files = set(outer_test_meta.file)
        train_rows = data[data.file.isin(train_files)].copy()
        calibration_rows = data[data.file.isin(calibration_files)].copy()
        test_rows = data[data.file.isin(test_files)].copy()

        detector, selected, threshold = fit_detector(
            train_rows, calibration_rows, feature_names, top_k, seed + fold
        )
        fault_model, fault_classes = fit_fault_model(train_rows, feature_names, seed + fold)
        det_predictions = aggregate_detector(test_rows, detector.predict_proba(test_rows[selected])[:, 1])
        fault_predictions = aggregate_fault(test_rows, fault_model, feature_names, fault_classes)
        cascade = form_cascade(det_predictions, fault_predictions, threshold)
        cascade["fold"] = fold
        all_predictions.append(cascade)
        metrics = summarize_cascade(cascade, class_names)
        metrics.update({
            "fold": fold,
            "train_recordings": int(len(train_files)),
            "calibration_recordings": int(len(calibration_files)),
            "test_recordings": int(len(test_files)),
            "detector_threshold": threshold,
        })
        fold_summaries.append(metrics)
        print(f"Fold {fold}/{n_splits}: cascade macro-F1={metrics['macro_f1']:.4f}, "
              f"balanced accuracy={metrics['balanced_accuracy']:.4f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_frame = pd.DataFrame(fold_summaries)
    pooled = summarize_cascade(predictions, class_names)
    result = {
        "evaluation": f"{n_splits}-fold stratified CV grouped by original WAV; each outer fold has an inner normal-only threshold calibration split",
        "outer_fold_mean": {key: float(fold_frame[key].mean()) for key in
                            ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")},
        "outer_fold_std": {key: float(fold_frame[key].std(ddof=1)) for key in
                           ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")},
        "pooled_oof_metrics": pooled,
        "classes": class_names,
        "note": "Threshold is calibrated on the inner validation normal WAVs only. RF is trained on abnormal inner-training WAVs. Every original WAV appears in exactly one outer test fold.",
    }
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "grouped_cv_oof_predictions.csv", index=False)
    fold_frame.to_csv(output / "grouped_cv_fold_metrics.csv", index=False)
    cm = confusion_matrix(predictions.true_output, predictions.predicted_output, labels=class_names)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output / "grouped_cv_confusion_matrix.csv")
    (output / "grouped_cv_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--split", type=Path, default=Path("car_diagnostics_results/recording_split.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("car_diagnostics_results"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/hybrid_cascade"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    features = [col for col in data.columns if col not in METADATA]
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    output = args.output
    holdout = evaluate_saved_holdout(data, features, args.split, output, args.model_root)
    cv = evaluate_nested_grouped_cv(data, features, output, args.folds, args.top_k, args.seed)
    print("Holdout cascade:", json.dumps(holdout["metrics"]))
    print("Grouped CV cascade:", json.dumps(cv["outer_fold_mean"]))
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
