#!/usr/bin/env python3
"""Tune feature ranking and XGBoost settings using train/validation subjects only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import entropy
from sklearn.feature_selection import f_classif
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier


def jsd_score(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    lo, hi = float(min(np.min(a), np.min(b))), float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo + hi) or hi - lo < 1e-15:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
    pb = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    mid = (pa + pb) / 2
    return float((entropy(pa, mid, base=2) + entropy(pb, mid, base=2)) / 2)


def rank_features(X: pd.DataFrame, y: np.ndarray, names: list[str], method: str) -> list[str]:
    if method == "ALL":
        return names.copy()
    if method == "ANOVA":
        scores, _ = f_classif(X[names], y)
        values = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        values = np.array([jsd_score(X.loc[y == 0, c].to_numpy(), X.loc[y == 1, c].to_numpy()) for c in names])
    return [names[i] for i in np.argsort(-values, kind="stable")]


def make_model(y: np.ndarray, seed: int, depth: int, weighted: bool) -> XGBClassifier:
    ratio = (int(np.sum(y == 0)) / max(int(np.sum(y == 1)), 1)) if weighted else 1.0
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=ratio,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )


def best_balanced_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    scores = (tpr + 1.0 - fpr) / 2.0
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][int(np.argmax(scores[finite]))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--pipeline-metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/train_val_test/tuning"))
    parser.add_argument("--train-subjects", nargs="+", default=["id_00", "id_02"])
    parser.add_argument("--validation-subject", default="id_04")
    parser.add_argument("--top-k", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Filter before feature processing: test subject id_06 is never passed to scoring or fitting.
    frame = pd.read_csv(args.input)
    train_frame = frame[frame.subject.isin(args.train_subjects)].copy()
    val_frame = frame[frame.subject == args.validation_subject].copy()
    if train_frame.empty or val_frame.empty:
        raise ValueError("Training and validation subjects must exist in the input")
    meta = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    names = [c for c in train_frame.columns if c not in meta]
    if set(train_frame.subject.unique()) & set(val_frame.subject.unique()):
        raise ValueError("Training and validation subjects overlap")

    def xy(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return (part[names].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32),
                (part.label.to_numpy() == "abnormal").astype(np.int8))

    X_train, y_train = xy(train_frame)
    X_val, y_val = xy(val_frame)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both train and validation must contain normal and abnormal windows")

    selections = [(method, k) for method in ("ANOVA", "JSD") for k in sorted(set(args.top_k))]
    selections.append(("ALL", len(names)))
    rows: list[dict] = []
    candidate_features: dict[str, list[str]] = {}
    candidate_models: dict[str, XGBClassifier] = {}
    candidate_probabilities: dict[str, np.ndarray] = {}
    for method, k in selections:
        ranking = rank_features(X_train, y_train, names, method)
        selected = ranking if method == "ALL" else ranking[:k]
        for depth in (2, 4):
            for weighted in (False, True):
                key = f"{method}_k{k}_depth{depth}_weighted{weighted}"
                model = make_model(y_train, args.seed, depth, weighted)
                model.fit(X_train[selected], y_train)
                probabilities = model.predict_proba(X_val[selected])[:, 1]
                threshold = best_balanced_threshold(y_val, probabilities)
                predictions_half = (probabilities >= 0.5).astype(np.int8)
                predictions_best = (probabilities >= threshold).astype(np.int8)
                rows.append({
                    "candidate": key, "method": method, "top_k": k, "max_depth": depth,
                    "class_weighting": weighted, "validation_roc_auc": roc_auc_score(y_val, probabilities),
                    "validation_balanced_accuracy_at_0_5": balanced_accuracy_score(y_val, predictions_half),
                    "validation_balanced_accuracy_tuned_threshold": balanced_accuracy_score(y_val, predictions_best),
                    "validation_threshold": threshold,
                    "validation_precision_abnormal": precision_score(y_val, predictions_best, zero_division=0),
                    "validation_recall_abnormal": recall_score(y_val, predictions_best, zero_division=0),
                    "validation_f1_abnormal": f1_score(y_val, predictions_best, zero_division=0),
                })
                candidate_features[key] = selected
                candidate_models[key] = model
                candidate_probabilities[key] = probabilities
        print(f"Tuned {method} top-{k} candidates on validation subject {args.validation_subject}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        ["validation_roc_auc", "validation_balanced_accuracy_tuned_threshold"], ascending=False
    ).reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "validation_tuning_results.csv", index=False)
    best = results.iloc[0].to_dict()
    best_key = str(best["candidate"])
    threshold = float(best["validation_threshold"])
    selected_train_val = candidate_features[best_key]

    # Refit selector and model on train+validation; keep test completely out of this process.
    train_val = pd.concat([train_frame, val_frame], ignore_index=True)
    X_train_val, y_train_val = xy(train_val)
    method = str(best["method"])
    final_ranking = rank_features(X_train_val, y_train_val, names, method)
    final_k = int(best["top_k"])
    selected_final = final_ranking if method == "ALL" else final_ranking[:final_k]
    final_model = make_model(y_train_val, args.seed, int(best["max_depth"]), bool(best["class_weighting"]))
    final_model.fit(X_train_val[selected_final], y_train_val)

    model_dir = args.output / "tuned_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_model.json"
    final_model.save_model(model_path)
    pd.DataFrame({"feature": names, "rank": [final_ranking.index(n) + 1 for n in names]}).sort_values("rank").to_csv(
        model_dir / "selected_feature_ranking.csv", index=False
    )
    (model_dir / "selected_features.json").write_text(json.dumps(selected_final, indent=2), encoding="utf-8")
    pipeline = json.loads(args.pipeline_metadata.read_text(encoding="utf-8")) if args.pipeline_metadata.exists() else {}
    ratio = int(np.sum(y_train_val == 0)) / max(int(np.sum(y_train_val == 1)), 1) if bool(best["class_weighting"]) else 1.0
    metadata = {
        "model_file": model_path.name, "classifier": "XGBoost",
        "feature_selection": method, "top_k": len(selected_final),
        "selected_features_file": "selected_features.json", "selected_features": selected_final,
        "label_mapping": {"normal": 0, "abnormal": 1}, "prediction_threshold": threshold,
        "threshold_tuned_on": f"validation subject {args.validation_subject}",
        "fit_subjects": sorted(set(args.train_subjects) | {args.validation_subject}),
        "excluded_test_subject": "id_06", "xgboost_version": xgboost.__version__,
        "random_seed": args.seed,
        "model_parameters": {"n_estimators": 200, "max_depth": int(best["max_depth"]),
                             "learning_rate": 0.05, "subsample": 0.8,
                             "colsample_bytree": 0.8, "scale_pos_weight": ratio},
        "preprocessing_from_feature_extraction_run": pipeline,
        "validation_selected_configuration": best,
        "note": "This candidate was selected using validation only and refit on train+validation. Test data was excluded.",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "best_validation_configuration.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Best validation configuration: " + json.dumps(best, sort_keys=True), flush=True)
    print(f"Saved tuned candidate (fit on train+validation) to {model_dir}; test data was not used.", flush=True)


if __name__ == "__main__":
    main()
