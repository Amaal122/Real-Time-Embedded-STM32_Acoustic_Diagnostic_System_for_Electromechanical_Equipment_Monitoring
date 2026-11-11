#!/usr/bin/env python3
"""Fit final JSD/XGBoost models on all recordings, one per MIMII model ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy.stats import entropy
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier


def jsd_rank(X: pd.DataFrame, y: np.ndarray, names: list[str]) -> list[str]:
    scores = []
    for name in names:
        normal, abnormal = X.loc[y == 0, name].to_numpy(), X.loc[y == 1, name].to_numpy()
        low, high = float(min(normal.min(), abnormal.min())), float(max(normal.max(), abnormal.max()))
        if not np.isfinite(low + high) or high - low < 1e-15:
            scores.append(0.0)
            continue
        edges = np.linspace(low, high, 33)
        p = np.histogram(normal, bins=edges)[0].astype(float) + 1e-12
        q = np.histogram(abnormal, bins=edges)[0].astype(float) + 1e-12
        p, q = p / p.sum(), q / q.sum()
        mid = (p + q) / 2
        scores.append(float((entropy(p, mid, base=2) + entropy(q, mid, base=2)) / 2))
    return [names[i] for i in np.argsort(-np.asarray(scores), kind="stable")]


def threshold_from_oof(rows: pd.DataFrame) -> tuple[float, dict]:
    y = (rows.true_label.to_numpy() == "abnormal").astype(np.int8)
    p = rows.probability_abnormal.to_numpy()
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    balanced = (tpr + 1.0 - fpr) / 2.0
    threshold = float(thresholds[finite][int(np.argmax(balanced[finite]))])
    cv_at_half = float(balanced_accuracy_score(y, p >= 0.5))
    return threshold, {"oof_recordings": int(len(rows)), "roc_auc": float(roc_auc_score(y, p)),
                       "balanced_accuracy_at_0_5": cv_at_half,
                       "balanced_accuracy_at_threshold_selected_from_oof": float(balanced[finite].max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--pipeline-metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--oof", type=Path, default=Path("xgboost_comparison/per_model_id_cv/recording_oof_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/final_per_model_id"))
    parser.add_argument("--model-ids", nargs="+", default=["id_00", "id_02", "id_04", "id_06"])
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    oof = pd.read_csv(args.oof)
    pipeline = json.loads(args.pipeline_metadata.read_text(encoding="utf-8")) if args.pipeline_metadata.exists() else {}
    metadata_columns = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    features = [c for c in data.columns if c not in metadata_columns]
    if not 1 <= args.top_k <= len(features):
        raise ValueError(f"--top-k must be between 1 and {len(features)}")
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 200,
              "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "reg_lambda": 1.0, "scale_pos_weight": 1.0,
              "random_state": args.seed, "n_jobs": -1, "tree_method": "hist"}
    final_summary = []

    for model_id in args.model_ids:
        rows = data[data.subject == model_id].copy()
        oof_rows = oof[(oof.model_id == model_id) & (oof.method == "JSD")]
        if rows.empty or oof_rows.empty:
            raise ValueError(f"Missing feature or OOF rows for {model_id}")
        X = rows[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        y = (rows.label.to_numpy() == "abnormal").astype(np.int8)
        selected = jsd_rank(X, y, features)[:args.top_k]
        model = XGBClassifier(**params)
        model.fit(X[selected], y)
        threshold, oof_metrics = threshold_from_oof(oof_rows)

        model_dir = args.output / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "xgboost_model.json"
        model.save_model(model_path)
        (model_dir / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        model_metadata = {
            "model_file": model_path.name, "classifier": "final XGBoost per known MIMII model ID",
            "machine_model_id": model_id, "feature_selection": "JSD fit on all labelled WAV windows",
            "top_k": args.top_k, "selected_features_file": "selected_features.json",
            "selected_features": selected, "label_mapping": {"normal": 0, "abnormal": 1},
            "prediction_threshold": threshold,
            "threshold_selection": "maximize balanced accuracy on grouped 5-fold out-of-fold recording scores",
            "fit_recordings": int(rows.file.nunique()), "fit_windows": int(len(rows)),
            "xgboost_version": xgboost.__version__, "random_seed": args.seed,
            "model_parameters": params,
            "preprocessing_from_feature_extraction_run": pipeline,
            "oof_reference_metrics": oof_metrics,
            "note": "Final model refit on all recordings. OOF metrics are exploratory because the dataset was used in earlier experiments.",
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
        final_summary.append({"model_id": model_id, "fit_recordings": int(rows.file.nunique()),
                              "fit_windows": int(len(rows)), "selected_features": args.top_k,
                              "threshold_from_oof": threshold, **oof_metrics})
        print(f"Fit final model {model_id}: {rows.file.nunique()} WAV recordings, threshold={threshold:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_summary).to_csv(args.output / "final_model_summary.csv", index=False)
    print(f"Saved final per-ID models to {args.output}", flush=True)


if __name__ == "__main__":
    main()
