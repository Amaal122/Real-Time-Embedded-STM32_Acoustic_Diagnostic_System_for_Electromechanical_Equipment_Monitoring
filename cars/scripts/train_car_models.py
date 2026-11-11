#!/usr/bin/env python3
"""Train/evaluate car fault classifiers and a normal-only MLP autoencoder."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


METADATA = {"label", "condition", "fault_type", "class_in_state", "file", "window_start_sec", "sample_rate"}
SEED = 42


def jsd_score(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    low, high = float(min(np.min(a), np.min(b))), float(max(np.max(a), np.max(b)))
    if not np.isfinite(low + high) or high - low < 1e-15:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    p = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
    q = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2
    return float((entropy(p, middle, base=2) + entropy(q, middle, base=2)) / 2)


def rank_features(x: pd.DataFrame, y: np.ndarray, feature_names: list[str], method: str) -> pd.DataFrame:
    if method == "ANOVA":
        scores, pvals = f_classif(x[feature_names], y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    elif method == "JSD":
        classes = np.unique(y)
        scores = []
        for name in feature_names:
            class_values = [x.loc[y == cls, name].to_numpy() for cls in classes]
            pairs = [jsd_score(class_values[i], class_values[j])
                     for i in range(len(classes)) for j in range(i + 1, len(classes))]
            scores.append(float(np.mean(pairs)) if pairs else 0.0)
        scores = np.asarray(scores)
        pvals = np.full(len(feature_names), np.nan)
    else:
        raise ValueError(method)
    ranking = pd.DataFrame({"feature": feature_names, "score": scores, "p_value": pvals})
    ranking = ranking.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def make_xgb(task: str, seed: int = SEED) -> XGBClassifier:
    common = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    if task == "binary":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **common)


def aggregate_binary(rows: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    data = rows[["file", "label", "condition", "fault_type"]].copy()
    data["true_abnormal"] = (data.label == "abnormal").astype(int)
    data["score_abnormal"] = probabilities
    return data.groupby("file", sort=False).agg(
        label=("label", "first"), condition=("condition", "first"),
        fault_type=("fault_type", "first"), y_true=("true_abnormal", "first"),
        score=("score_abnormal", "mean"), windows=("score_abnormal", "size"),
    ).reset_index()


def binary_metrics(data: pd.DataFrame, threshold: float) -> dict:
    y = data.y_true.to_numpy(dtype=int)
    scores = data.score.to_numpy(dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recordings": int(len(data)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_abnormal": float(precision_score(y, pred, zero_division=0)),
        "recall_abnormal": float(recall_score(y, pred, zero_division=0)),
        "f1_abnormal": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibrate_upper_threshold(normal_scores: np.ndarray, target_fpr: float) -> float:
    # Higher quantile is conservative when the validation set is small or has tied scores.
    return float(np.quantile(normal_scores, 1.0 - target_fpr, method="higher"))


def aggregate_multiclass(rows: pd.DataFrame, probabilities: np.ndarray, class_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    data = rows[["file", "fault_type", "condition"]].copy().reset_index(drop=True)
    prob_cols = [f"p_{i}" for i in range(len(class_names))]
    probs = pd.DataFrame(probabilities, columns=prob_cols)
    data = pd.concat([data, probs], axis=1)
    result = data.groupby("file", sort=False).agg(
        fault_type=("fault_type", "first"), condition=("condition", "first"),
        **{col: (col, "mean") for col in prob_cols},
    ).reset_index()
    matrix = result[prob_cols].to_numpy()
    predictions = np.asarray(class_names, dtype=object)[np.argmax(matrix, axis=1)]
    result["predicted_fault_type"] = predictions
    return result, predictions


def multiclass_metrics(data: pd.DataFrame) -> dict:
    truth = data.fault_type.to_numpy()
    pred = data.predicted_fault_type.to_numpy()
    return {
        "recordings": int(len(data)),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "fault_classes": sorted(np.unique(truth).tolist()),
    }


def fit_selection_model(
    method: str,
    task: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: list[str],
    output: Path,
    top_k: int,
):
    ranking = rank_features(x_train, y_train, feature_names, method)
    selected = ranking.feature.head(min(top_k, len(feature_names))).tolist()
    ranking.to_csv(output / f"{task}_feature_ranking_{method.lower()}.csv", index=False)
    return make_xgb(task), selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results"))
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20,
                        help="Fraction of the train+validation recordings reserved for validation")
    parser.add_argument("--target-fpr", type=float, default=0.05,
                        help="Normal-only calibration false-alarm target used for both XGBoost and AE")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 0 < args.test_size < 0.5 or not 0 < args.validation_size < 0.5:
        parser.error("test-size and validation-size must be between 0 and 0.5")
    if not 0 < args.target_fpr < 0.5:
        parser.error("target-fpr must be between 0 and 0.5")

    data = pd.read_csv(args.input)
    features = [name for name in data.columns if name not in METADATA]
    if len(features) != 65:
        raise ValueError(f"Expected 65 features, found {len(features)}")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    # Split original recordings first; their overlapping windows always stay together.
    recordings = data.groupby("file", sort=True).agg(
        label=("label", "first"), fault_type=("fault_type", "first"), condition=("condition", "first")
    ).reset_index()
    train_val, test = train_test_split(
        recordings, test_size=args.test_size, random_state=args.seed, stratify=recordings.fault_type
    )
    train, validation = train_test_split(
        train_val, test_size=args.validation_size, random_state=args.seed + 1,
        stratify=train_val.fault_type,
    )
    split_files = {
        "train": set(train.file), "validation": set(validation.file), "test": set(test.file)
    }
    if split_files["train"] & split_files["validation"] or split_files["train"] & split_files["test"] or split_files["validation"] & split_files["test"]:
        raise AssertionError("Recording-level split overlap detected")
    args.output.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for split_name, split_frame in (("train", train), ("validation", validation), ("test", test)):
        temp = split_frame.copy()
        temp["split"] = split_name
        split_rows.append(temp)
    pd.concat(split_rows).sort_values(["split", "file"]).to_csv(args.output / "recording_split.csv", index=False)
    windows = {name: data[data.file.isin(files)].copy() for name, files in split_files.items()}
    summary = []
    split_metadata = {name: int(len(files)) for name, files in split_files.items()}
    methods = ("JSD", "ANOVA")
    binary_models: dict[str, tuple[XGBClassifier, list[str], float]] = {}

    # Supervised binary detection; each model's threshold is calibrated on normal validation WAVs only.
    binary_summaries = []
    for method in methods:
        train_rows, val_rows, test_rows = windows["train"], windows["validation"], windows["test"]
        y_train = (train_rows.label == "abnormal").astype(int).to_numpy()
        model, selected = fit_selection_model(method, "binary", train_rows[features], y_train,
                                               features, args.output, args.top_k)
        model.fit(train_rows[selected], y_train,
                  sample_weight=compute_sample_weight("balanced", y_train))
        val_predictions = aggregate_binary(val_rows, model.predict_proba(val_rows[selected])[:, 1])
        normal_val_scores = val_predictions.loc[val_predictions.y_true == 0, "score"].to_numpy()
        threshold = calibrate_upper_threshold(normal_val_scores, args.target_fpr)
        binary_models[method] = (model, selected, threshold)
        test_predictions = aggregate_binary(test_rows, model.predict_proba(test_rows[selected])[:, 1])
        val_metrics = binary_metrics(val_predictions, threshold)
        test_metrics = binary_metrics(test_predictions, threshold)
        val_predictions["predicted"] = np.where(val_predictions.score >= threshold, "abnormal", "normal")
        test_predictions["predicted"] = np.where(test_predictions.score >= threshold, "abnormal", "normal")
        val_predictions.to_csv(args.output / f"binary_validation_predictions_{method.lower()}.csv", index=False)
        test_predictions.to_csv(args.output / f"binary_test_predictions_{method.lower()}.csv", index=False)
        model.save_model(args.output / f"xgboost_binary_{method.lower()}.json")
        (args.output / f"xgboost_binary_{method.lower()}_metadata.json").write_text(json.dumps({
            "method": method, "selected_features": selected, "threshold": threshold,
            "threshold_calibration": f"{1-args.target_fpr:.0%} quantile of normal validation recording scores",
            "feature_count": len(selected), "seed": args.seed,
        }, indent=2), encoding="utf-8")
        binary_summaries.append({"model": f"XGBoost {method}", "task": "normal_vs_abnormal",
                                 **{f"validation_{k}": v for k, v in val_metrics.items()},
                                 **{f"test_{k}": v for k, v in test_metrics.items()}})
        summary.append({"model": f"XGBoost {method}", "task": "normal_vs_abnormal", **test_metrics})

    pd.DataFrame(binary_summaries).to_csv(args.output / "binary_xgboost_summary.csv", index=False)

    # Normal-only reconstruction autoencoder. Fit scaling and network on training normal WAVs only.
    normal_train = windows["train"][windows["train"].label == "normal"]
    scaler = StandardScaler().fit(normal_train[features])
    x_normal_train = scaler.transform(normal_train[features])
    ae = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=64, learning_rate_init=1e-3, max_iter=700,
        early_stopping=False, n_iter_no_change=30, random_state=args.seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ae.fit(x_normal_train, x_normal_train)
    ae_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    def reconstruction_by_file(rows: pd.DataFrame) -> pd.DataFrame:
        scaled = scaler.transform(rows[features])
        reconstructed = ae.predict(scaled)
        error = np.mean((scaled - reconstructed) ** 2, axis=1)
        temp = rows[["file", "label", "condition", "fault_type"]].copy()
        temp["error"] = error
        temp["y_true"] = (temp.label == "abnormal").astype(int)
        return temp.groupby("file", sort=False).agg(
            label=("label", "first"), condition=("condition", "first"),
            fault_type=("fault_type", "first"), y_true=("y_true", "first"),
            score=("error", "mean"), windows=("error", "size"),
        ).reset_index()

    ae_val = reconstruction_by_file(windows["validation"])
    ae_threshold = calibrate_upper_threshold(
        ae_val.loc[ae_val.y_true == 0, "score"].to_numpy(), args.target_fpr
    )
    ae_test = reconstruction_by_file(windows["test"])
    ae_val_metrics = binary_metrics(ae_val, ae_threshold)
    ae_test_metrics = binary_metrics(ae_test, ae_threshold)
    ae_val["predicted"] = np.where(ae_val.score >= ae_threshold, "abnormal", "normal")
    ae_test["predicted"] = np.where(ae_test.score >= ae_threshold, "abnormal", "normal")
    ae_val.to_csv(args.output / "ae_validation_predictions.csv", index=False)
    ae_test.to_csv(args.output / "ae_test_predictions.csv", index=False)
    joblib.dump({"scaler": scaler, "model": ae, "features": features,
                 "threshold": ae_threshold, "threshold_method": f"normal validation {1-args.target_fpr:.0%} quantile"},
                args.output / "normal_only_autoencoder.joblib")
    ae_result = {"model": "Normal-only MLP autoencoder", "task": "normal_vs_abnormal",
                 **{f"validation_{k}": v for k, v in ae_val_metrics.items()},
                 **{f"test_{k}": v for k, v in ae_test_metrics.items()},
                 "training_normal_windows": int(len(normal_train)), "iterations": int(ae.n_iter_),
                 "convergence_warning": bool(ae_warning)}
    (args.output / "autoencoder_metrics.json").write_text(json.dumps(ae_result, indent=2), encoding="utf-8")
    summary.append({"model": "Normal-only MLP autoencoder", "task": "normal_vs_abnormal", **ae_test_metrics})

    # Multiclass fault diagnosis, run only on WAVs labelled abnormal.
    train_fault = windows["train"][windows["train"].label == "abnormal"]
    val_fault = windows["validation"][windows["validation"].label == "abnormal"]
    test_fault = windows["test"][windows["test"].label == "abnormal"]
    encoder = LabelEncoder().fit(train_fault.fault_type)
    y_fault_train = encoder.transform(train_fault.fault_type)
    diagnosis_summaries = []
    fault_models: dict[str, tuple[XGBClassifier, list[str]]] = {}
    for method in methods:
        model, selected = fit_selection_model(method, "fault_multiclass", train_fault[features],
                                               y_fault_train, features, args.output, args.top_k)
        model.fit(train_fault[selected], y_fault_train,
                  sample_weight=compute_sample_weight("balanced", y_fault_train))
        fault_models[method] = (model, selected)
        val_row_probs = model.predict_proba(val_fault[selected])
        test_row_probs = model.predict_proba(test_fault[selected])
        val_agg, _ = aggregate_multiclass(val_fault, val_row_probs, encoder.classes_.tolist())
        test_agg, _ = aggregate_multiclass(test_fault, test_row_probs, encoder.classes_.tolist())
        val_metrics = multiclass_metrics(val_agg)
        test_metrics = multiclass_metrics(test_agg)
        val_agg.to_csv(args.output / f"fault_validation_predictions_{method.lower()}.csv", index=False)
        test_agg.to_csv(args.output / f"fault_test_predictions_{method.lower()}.csv", index=False)
        cm = confusion_matrix(test_agg.fault_type, test_agg.predicted_fault_type, labels=encoder.classes_)
        pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            args.output / f"fault_test_confusion_{method.lower()}.csv")
        model.save_model(args.output / f"xgboost_fault_multiclass_{method.lower()}.json")
        (args.output / f"xgboost_fault_multiclass_{method.lower()}_metadata.json").write_text(json.dumps({
            "method": method, "task": "fault type classification among abnormal clips only",
            "selected_features": selected, "class_names": encoder.classes_.tolist(),
            "feature_count": len(selected), "seed": args.seed,
        }, indent=2), encoding="utf-8")
        diagnosis_summaries.append({"model": f"XGBoost fault diagnosis {method}",
                                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                                    **{f"test_{k}": v for k, v in test_metrics.items()}})
        summary.append({"model": f"XGBoost fault diagnosis {method}", "task": "fault_type_multiclass", **test_metrics})
    pd.DataFrame(diagnosis_summaries).to_csv(args.output / "fault_diagnosis_summary.csv", index=False)

    # End-to-end cascade: binary gate, then fault subtype only when flagged abnormal.
    cascade_summaries = []
    cascade_class_names = ["normal"] + encoder.classes_.tolist()
    for method in methods:
        binary_model, binary_features, threshold = binary_models[method]
        fault_model, fault_features = fault_models[method]
        for split_name in ("validation", "test"):
            rows = windows[split_name]
            binary = aggregate_binary(rows, binary_model.predict_proba(rows[binary_features])[:, 1])
            binary["stage1_prediction"] = np.where(binary.score >= threshold, "abnormal", "normal")
            subtype, _ = aggregate_multiclass(
                rows, fault_model.predict_proba(rows[fault_features]), encoder.classes_.tolist()
            )
            combined = binary[["file", "label", "fault_type", "stage1_prediction"]].merge(
                subtype[["file", "predicted_fault_type"]], on="file", validate="one_to_one"
            )
            combined["true_diagnosis"] = np.where(
                combined.label == "normal", "normal", combined.fault_type
            )
            combined["predicted_diagnosis"] = np.where(
                combined.stage1_prediction == "normal", "normal", combined.predicted_fault_type
            )
            truth = combined.true_diagnosis.to_numpy()
            predicted = combined.predicted_diagnosis.to_numpy()
            cm = confusion_matrix(truth, predicted, labels=cascade_class_names)
            result = {
                "model": f"XGBoost {method} cascade",
                "task": "normal_detection_then_fault_type",
                "recordings": int(len(combined)),
                "accuracy": float(accuracy_score(truth, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                "macro_f1": float(f1_score(truth, predicted, labels=cascade_class_names,
                                            average="macro", zero_division=0)),
                "binary_threshold": float(threshold),
                "fault_classes": cascade_class_names,
            }
            combined.to_csv(args.output / f"cascade_{split_name}_predictions_{method.lower()}.csv", index=False)
            if split_name == "test":
                pd.DataFrame(cm, index=cascade_class_names, columns=cascade_class_names).to_csv(
                    args.output / f"cascade_test_confusion_{method.lower()}.csv"
                )
                cascade_summaries.append(result)
                summary.append(result)
    pd.DataFrame(cascade_summaries).to_csv(args.output / "cascade_summary.csv", index=False)

    pd.DataFrame(summary).to_csv(args.output / "comparison_summary.csv", index=False)
    split_info = {
        "seed": args.seed,
        "recordings_per_split": split_metadata,
        "window_rows_per_split": {name: int(len(rows)) for name, rows in windows.items()},
        "file_overlap": False,
        "target_false_positive_rate": args.target_fpr,
        "threshold_rule": "95th percentile of normal validation recording scores for each XGBoost and AE detector",
        "selector_top_k": min(args.top_k, len(features)),
        "selectors": list(methods),
        "fault_diagnosis": "10-way fault-type classification on abnormal recordings only; normal clips are excluded",
        "autoencoder": "scikit-learn MLPRegressor used as a reconstruction autoencoder because TensorFlow and PyTorch are unavailable",
        "limitations": [
            "Single stratified recording-level holdout; estimates are uncertain, especially for the smallest fault classes.",
            "The autoencoder is a lightweight MLP reconstruction baseline, not the official car-dataset baseline.",
            "The dataset has three operating conditions; pooled results may reflect condition as well as fault acoustics.",
            "A validation-normal quantile targets a calibration false-alarm rate; the realized test rate may differ.",
        ],
    }
    (args.output / "evaluation_metadata.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print("Recording split:", json.dumps(split_metadata))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
