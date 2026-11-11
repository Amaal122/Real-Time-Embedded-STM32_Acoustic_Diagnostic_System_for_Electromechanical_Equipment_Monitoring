#!/usr/bin/env python3
"""Fit and save the final XGBoost model using the top ANOVA features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("features_0dB/features_65.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("features_0dB/run_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/final_model"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metadata_cols = {"label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"}
    features = [col for col in frame.columns if col not in metadata_cols]
    if set(frame.label.unique()) != {"normal", "abnormal"}:
        raise ValueError("Training data must contain both normal and abnormal classes")
    if not 1 <= args.top_k <= len(features):
        raise ValueError(f"--top-k must be between 1 and {len(features)}")

    X = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = (frame.label.to_numpy() == "abnormal").astype(np.int8)
    f_values, p_values = f_classif(X, y)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-f_values, kind="stable")
    selected = [features[i] for i in order[:args.top_k]]

    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=negatives / positives,
        random_state=args.seed, n_jobs=-1, tree_method="hist",
    )
    model.fit(X[selected], y)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "xgboost_model.json"
    model.save_model(model_path)
    ranking = pd.DataFrame({"feature": features, "anova_f": f_values,
                            "anova_p": np.nan_to_num(p_values, nan=1.0)})
    ranking["rank"] = ranking.anova_f.rank(ascending=False, method="min").astype(int)
    ranking.sort_values("rank").to_csv(args.output / "anova_feature_ranking.csv", index=False)
    (args.output / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    pipeline = {}
    if args.metadata.exists():
        pipeline = json.loads(args.metadata.read_text(encoding="utf-8"))
    bundle_metadata = {
        "model_file": model_path.name,
        "classifier": "XGBoost",
        "feature_selection": "ANOVA F statistic, fit on all available training rows",
        "selected_features_file": "selected_features.json",
        "selected_features": selected,
        "label_mapping": {"normal": 0, "abnormal": 1},
        "prediction_threshold": 0.5,
        "training_windows": len(frame),
        "training_subjects": sorted(frame.subject.unique().tolist()),
        "training_class_counts": {str(k): int(v) for k, v in frame.label.value_counts().to_dict().items()},
        "xgboost_version": xgboost.__version__,
        "random_seed": args.seed,
        "model_parameters": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                             "subsample": 0.8, "colsample_bytree": 0.8,
                             "scale_pos_weight": negatives / positives},
        "preprocessing_from_feature_extraction_run": pipeline,
        "training_performance_warning": "This model was fitted on all rows; use the separate subject-held-out CV results to estimate performance.",
    }
    (args.output / "model_metadata.json").write_text(json.dumps(bundle_metadata, indent=2), encoding="utf-8")
    print(f"Trained on {len(frame)} windows; selected {len(selected)} ANOVA features.", flush=True)
    print(f"Saved model bundle to {args.output}", flush=True)
    print("Selected features: " + ", ".join(selected), flush=True)


if __name__ == "__main__":
    main()
