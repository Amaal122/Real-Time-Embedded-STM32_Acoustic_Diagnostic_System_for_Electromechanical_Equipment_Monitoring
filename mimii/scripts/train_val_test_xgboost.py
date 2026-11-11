#!/usr/bin/env python3
"""Train, tune a probability threshold on validation, and evaluate by subject."""

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
from xgboost import XGBClassifier


def select_features(X: pd.DataFrame, y: np.ndarray, names: list[str], top_k: int, method: str) -> tuple[list[str], pd.DataFrame]:
    if method == "ANOVA":
        scores, p_values = f_classif(X[names], y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
        ranking = pd.DataFrame({"feature": names, "anova_f": scores, "anova_p": p_values})
        score_column = "anova_f"
    else:
        scores = []
        for name in names:
            a, b = X.loc[y == 0, name].to_numpy(), X.loc[y == 1, name].to_numpy()
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
        scores = np.asarray(scores)
        ranking = pd.DataFrame({"feature": names, "jsd_score": scores})
        score_column = "jsd_score"
    order = np.argsort(-scores, kind="stable")
    ranking["rank"] = ranking[score_column].rank(ascending=False, method="min").astype(int)
    return [names[i] for i in order[:top_k]], ranking.sort_values("rank")


def make_model(y: np.ndarray, seed: int) -> XGBClassifier:
    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=negatives / positives,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )


def metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "n_windows": len(y), "accuracy": float(accuracy_score(y, predictions)),
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
    parser.add_argument("--pipeline-metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/train_val_test"))
    parser.add_argument("--train-subjects", nargs="+", default=["id_00", "id_02"])
    parser.add_argument("--validation-subject", default="id_04")
    parser.add_argument("--test-subject", default="id_06")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--selection-method", choices=["ANOVA", "JSD"], default="ANOVA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = set(args.train_subjects) | {args.validation_subject, args.test_subject}
    if len(splits) != len(args.train_subjects) + 2:
        parser.error("Train, validation, and test subject IDs must be distinct")
    frame = pd.read_csv(args.input)
    available = set(frame.subject.unique())
    missing = splits - available
    if missing:
        parser.error(f"Subject IDs not found in CSV: {sorted(missing)}")
    metadata_cols = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    names = [col for col in frame.columns if col not in metadata_cols]
    if not 1 <= args.top_k <= len(names):
        parser.error(f"--top-k must be between 1 and {len(names)}")

    train_frame = frame[frame.subject.isin(args.train_subjects)].copy()
    val_frame = frame[frame.subject == args.validation_subject].copy()
    test_frame = frame[frame.subject == args.test_subject].copy()

    def xy(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return (part[names].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32),
                (part.label.to_numpy() == "abnormal").astype(np.int8))

    X_train, y_train = xy(train_frame)
    X_val, y_val = xy(val_frame)
    X_test, y_test = xy(test_frame)
    for split_name, labels in (("train", y_train), ("validation", y_val), ("test", y_test)):
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{split_name} split must contain both classes")

    # Use train only for feature selection and fitting; validation selects the threshold.
    selected_train, _ = select_features(X_train, y_train, names, args.top_k, args.selection_method)
    validation_model = make_model(y_train, args.seed)
    validation_model.fit(X_train[selected_train], y_train)
    val_probabilities = validation_model.predict_proba(X_val[selected_train])[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, val_probabilities)
    finite = np.isfinite(thresholds)
    best_threshold_index = int(np.argmax(((tpr + 1.0 - fpr) / 2.0)[finite]))
    validation_threshold = float(thresholds[finite][best_threshold_index])
    val_metrics = metrics(y_val, val_probabilities, validation_threshold)

    # Refit feature ranking and classifier on train+validation, then touch the test split once.
    train_val_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    X_train_val, y_train_val = xy(train_val_frame)
    selected_final, final_ranking = select_features(X_train_val, y_train_val, names, args.top_k, args.selection_method)
    final_model = make_model(y_train_val, args.seed)
    final_model.fit(X_train_val[selected_final], y_train_val)
    test_probabilities = final_model.predict_proba(X_test[selected_final])[:, 1]
    test_metrics = metrics(y_test, test_probabilities, validation_threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "xgboost_model.json"
    final_model.save_model(final_model_path)
    final_ranking.to_csv(model_dir / "anova_feature_ranking.csv", index=False)
    (model_dir / "selected_features.json").write_text(json.dumps(selected_final, indent=2), encoding="utf-8")

    pipeline = json.loads(args.pipeline_metadata.read_text(encoding="utf-8")) if args.pipeline_metadata.exists() else {}
    model_metadata = {
        "model_file": final_model_path.name, "classifier": "XGBoost",
        "feature_selection": f"{args.selection_method} fit on train+validation rows",
        "selected_features_file": "selected_features.json", "selected_features": selected_final,
        "label_mapping": {"normal": 0, "abnormal": 1},
        "prediction_threshold": validation_threshold,
        "threshold_tuned_on": "validation subject only; maximize balanced accuracy",
        "fit_subjects": sorted(train_val_frame.subject.unique().tolist()),
        "held_out_test_subject": args.test_subject,
        "xgboost_version": xgboost.__version__, "random_seed": args.seed,
        "model_parameters": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                             "subsample": 0.8, "colsample_bytree": 0.8,
                             "scale_pos_weight": int(np.sum(y_train_val == 0)) / max(int(np.sum(y_train_val == 1)), 1)},
        "preprocessing_from_feature_extraction_run": pipeline,
        "note": "Test subject was excluded from model fitting and threshold tuning. Earlier exploratory CV included all subjects, so this is a retrospective holdout, not a pristine untouched test.",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    (args.output / "split_metadata.json").write_text(json.dumps({
        "train_subjects": sorted(args.train_subjects), "validation_subject": args.validation_subject,
        "test_subject": args.test_subject,
        "window_counts": {"train": len(train_frame), "validation": len(val_frame), "test": len(test_frame)},
        "class_counts": {name: part.label.value_counts().to_dict() for name, part in
                         (("train", train_frame), ("validation", val_frame), ("test", test_frame))},
        "validation_threshold": validation_threshold,
        "feature_selection_refit_on_train_plus_validation_for_final_model": True,
        "previous_exploration_used_all_subjects": True,
    }, indent=2), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps({
        "validation_at_selected_threshold": val_metrics,
        "test_at_validation_selected_threshold": test_metrics,
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"probability_abnormal": test_probabilities,
                  "predicted_label": np.where(test_probabilities >= validation_threshold, "abnormal", "normal"),
                  "true_label": np.where(y_test == 1, "abnormal", "normal"),
                  "subject": test_frame.subject.to_numpy(), "file": test_frame.file.to_numpy(),
                  "window_start_sec": test_frame.window_start_sec.to_numpy()}).to_csv(args.output / "test_predictions.csv", index=False)

    print(f"Train: {len(train_frame)} windows, subjects {sorted(args.train_subjects)}")
    print(f"Validation: {len(val_frame)} windows, subject {args.validation_subject}; selected threshold {validation_threshold:.4f}")
    print(f"Test: {len(test_frame)} windows, subject {args.test_subject}")
    print("Test metrics: " + json.dumps(test_metrics, sort_keys=True), flush=True)
    print(f"Saved split evaluation and final model to {args.output}", flush=True)


if __name__ == "__main__":
    main()
