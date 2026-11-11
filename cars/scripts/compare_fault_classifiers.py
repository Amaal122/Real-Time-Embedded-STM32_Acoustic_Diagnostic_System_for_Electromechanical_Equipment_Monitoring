#!/usr/bin/env python3
"""Select a non-XGBoost model for multiclass fault diagnosis using validation WAVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from train_car_models import METADATA, rank_features


def aggregate_scores(rows: pd.DataFrame, model, features: list[str], class_names: list[str]) -> pd.DataFrame:
    x = rows[features]
    if hasattr(model, "decision_function") and isinstance(model, Pipeline):
        # Average scaled one-vs-rest decision scores across the two windows of each WAV.
        scores = model.decision_function(x)
    else:
        scores = model.predict_proba(x)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    pcols = [f"score_{i}" for i in range(len(class_names))]
    score_frame = pd.DataFrame(scores, columns=pcols)
    meta = rows[["file", "fault_type", "condition"]].reset_index(drop=True)
    combined = pd.concat([meta, score_frame], axis=1)
    result = combined.groupby("file", sort=False).agg(
        fault_type=("fault_type", "first"), condition=("condition", "first"),
        **{col: (col, "mean") for col in pcols},
    ).reset_index()
    result["predicted_fault_type"] = np.asarray(class_names, dtype=object)[
        np.argmax(result[pcols].to_numpy(), axis=1)
    ]
    return result


def score_metrics(data: pd.DataFrame) -> dict:
    y, pred = data.fault_type.to_numpy(), data.predicted_fault_type.to_numpy()
    return {
        "recordings": int(len(data)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def make_model(name: str, params: dict, seed: int):
    if name == "SVC_RBF":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced", decision_function_shape="ovr",
                           probability=False, random_state=seed, **params)),
        ])
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced",
                                      n_jobs=-1, random_state=seed, **params)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced",
                                    n_jobs=-1, random_state=seed, **params)
    raise ValueError(name)


def evaluate_variant(
    name: str, params: dict, selector: str, train_rows: pd.DataFrame, val_rows: pd.DataFrame,
    features: list[str], encoder: LabelEncoder, top_k: int, seed: int,
) -> tuple[dict, object, list[str]]:
    y_train = encoder.transform(train_rows.fault_type)
    ranking = rank_features(train_rows[features], y_train, features, selector) if selector != "ALL" else None
    chosen = features if ranking is None else ranking.feature.head(min(top_k, len(features))).tolist()
    model = make_model(name, params, seed)
    model.fit(train_rows[chosen], y_train)
    val = aggregate_scores(val_rows, model, chosen, encoder.classes_.tolist())
    result = {"model": name, "selector": selector, "parameters": json.dumps(params, sort_keys=True),
              **score_metrics(val)}
    return result, model, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--split", type=Path, default=Path("car_diagnostics_results/recording_split.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/fault_model_search"))
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    features = [c for c in data.columns if c not in METADATA]
    split = pd.read_csv(args.split)
    train_files = set(split.loc[split.split == "train", "file"])
    val_files = set(split.loc[split.split == "validation", "file"])
    test_files = set(split.loc[split.split == "test", "file"])
    train = data[data.file.isin(train_files) & (data.label == "abnormal")].copy()
    val = data[data.file.isin(val_files) & (data.label == "abnormal")].copy()
    test = data[data.file.isin(test_files) & (data.label == "abnormal")].copy()
    train_val = data[data.file.isin(train_files | val_files) & (data.label == "abnormal")].copy()
    encoder = LabelEncoder().fit(train.fault_type)

    variants = []
    for c in (0.1, 1.0, 10.0, 100.0):
        for selector in ("ALL", "ANOVA", "JSD"):
            variants.append(("SVC_RBF", {"C": c, "gamma": "scale"}, selector))
    for model_name in ("RandomForest", "ExtraTrees"):
        for leaf in (1, 3, 5):
            for selector in ("ALL", "ANOVA"):
                variants.append((model_name, {"min_samples_leaf": leaf}, selector))

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    fitted = []
    for model_name, params, selector in variants:
        result, model, chosen = evaluate_variant(model_name, params, selector, train, val,
                                                  features, encoder, args.top_k, args.seed)
        results.append(result)
        fitted.append((result, model, chosen, model_name, params, selector))
        print(f"Validation: {model_name} {selector} {params}: macro-F1={result['macro_f1']:.4f}, "
              f"balanced-accuracy={result['balanced_accuracy']:.4f}", flush=True)

    search = pd.DataFrame(results).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"], ascending=False
    ).reset_index(drop=True)
    search.to_csv(args.output / "validation_model_comparison.csv", index=False)
    best_row = search.iloc[0]
    best = next(item for item in fitted
                if item[0]["model"] == best_row.model
                and item[0]["selector"] == best_row.selector
                and item[0]["parameters"] == best_row.parameters)
    _, _, _, best_name, best_params, best_selector = best

    # Lock the chosen variant from validation, refit on train+validation, then evaluate test once.
    y_train_val = encoder.transform(train_val.fault_type)
    ranking = rank_features(train_val[features], y_train_val, features, best_selector) if best_selector != "ALL" else None
    selected = features if ranking is None else ranking.feature.head(min(args.top_k, len(features))).tolist()
    if ranking is not None:
        ranking.to_csv(args.output / f"selected_fault_features_{best_selector.lower()}.csv", index=False)
    final_model = make_model(best_name, best_params, args.seed)
    final_model.fit(train_val[selected], y_train_val)
    test_predictions = aggregate_scores(test, final_model, selected, encoder.classes_.tolist())
    metrics = score_metrics(test_predictions)
    test_predictions.to_csv(args.output / "selected_model_test_predictions.csv", index=False)
    joblib.dump({"model": final_model, "features": selected,
                 "classes": encoder.classes_.tolist(), "model_type": best_name,
                 "selector": best_selector, "parameters": best_params},
                args.output / "selected_fault_classifier.joblib")
    summary = {
        "selected_on": "validation macro-F1, tie-broken by balanced accuracy then accuracy",
        "selected_model": best_name,
        "selector": best_selector,
        "parameters": best_params,
        "selected_feature_count": len(selected),
        "test_metrics": metrics,
        "test_recordings": int(test_predictions.file.nunique()),
        "note": "Test set was not used to choose the model; test results are retrospective alongside previous project comparisons.",
    }
    (args.output / "selected_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTop validation candidates:")
    print(search.head(10).to_string(index=False))
    print("\nSelected and refit model test metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
