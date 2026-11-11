#!/usr/bin/env python3
"""Train one car RF classifier for normal and fault labels, with a 6-second clip decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


METADATA_COLUMNS = {
    "label", "condition", "fault_type", "class_in_state", "file",
    "window_start_sec", "sample_rate",
}


def diagnosis_labels(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame.label.to_numpy() == "normal", "normal", frame.fault_type.to_numpy())


def split_rows(frame: pd.DataFrame, split: pd.DataFrame, split_name: str) -> pd.DataFrame:
    files = set(split.loc[split.split == split_name, "file"].astype(str))
    return frame[frame.file.astype(str).isin(files)].copy()


def recording_predictions(rows: pd.DataFrame, model: RandomForestClassifier,
                         features: list[str]) -> pd.DataFrame:
    probabilities = model.predict_proba(rows[features])
    return pd.DataFrame({
        "file": rows.file.astype(str).to_numpy(),
        "truth": rows.diagnosis.to_numpy(),
        "prediction": np.asarray(model.classes_)[probabilities.argmax(axis=1)],
        "window_count": rows.window_count.to_numpy(),
    })


def aggregate_recordings(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Make one recording-level example by averaging its overlapping window features."""
    aggregations = {
        "diagnosis": ("diagnosis", "first"),
        "label": ("label", "first"),
        "condition": ("condition", "first"),
        "fault_type": ("fault_type", "first"),
        "window_count": ("diagnosis", "size"),
        **{name: (name, "mean") for name in features},
    }
    return frame.groupby("file", sort=False).agg(**aggregations).reset_index()


def score_recordings(result: pd.DataFrame, classes: list[str]) -> dict:
    truth = result.truth.to_numpy()
    prediction = result.prediction.to_numpy()
    return {
        "recordings": int(len(result)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes,
                                    average="macro", zero_division=0)),
        "class_names": classes,
    }


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def export_c(model: RandomForestClassifier, output_dir: Path) -> dict:
    """Export a compact uint8 leaf-probability RF for window streaming on an MCU."""
    classes = [str(value) for value in model.classes_]
    all_features: list[int] = []
    all_thresholds: list[float] = []
    all_left: list[int] = []
    all_right: list[int] = []
    all_leaf_probabilities: list[list[int]] = []
    roots: list[int] = []
    maximum_depth = 0

    for estimator in model.estimators_:
        tree = estimator.tree_
        offset = len(all_features)
        roots.append(offset)
        maximum_depth = max(maximum_depth, int(tree.max_depth))
        values = tree.value[:, 0, :]
        for node in range(tree.node_count):
            feature = int(tree.feature[node])
            all_features.append(feature if feature >= 0 else -1)
            all_thresholds.append(float(tree.threshold[node]) if feature >= 0 else 0.0)
            left = int(tree.children_left[node])
            right = int(tree.children_right[node])
            all_left.append(offset + left if left >= 0 else -1)
            all_right.append(offset + right if right >= 0 else -1)
            if feature < 0:
                distribution = np.asarray(values[node], dtype=np.float64)
                total = float(distribution.sum())
                if total <= 0:
                    quantized = np.zeros(len(classes), dtype=np.uint8)
                else:
                    exact = distribution / total * 255.0
                    quantized = np.floor(exact).astype(np.int64)
                    remainder = 255 - int(quantized.sum())
                    if remainder > 0:
                        order = np.argsort(-(exact - quantized), kind="stable")
                        quantized[order[:remainder]] += 1
                    quantized = quantized.astype(np.uint8)
                all_leaf_probabilities.append(quantized.tolist())
            else:
                all_leaf_probabilities.append([0] * len(classes))

    node_count = len(all_features)
    n_classes = len(classes)
    if node_count >= 2**32:
        raise ValueError("Tree ensemble is too large for the generated uint32_t node indices")

    header = [
        "#ifndef CAR_GLOBAL_RF_MODEL_H", "#define CAR_GLOBAL_RF_MODEL_H", "",
        "#include <stddef.h>", "#include <stdint.h>", "",
        f"#define CAR_RF_FEATURE_COUNT {model.n_features_in_}u",
        f"#define CAR_RF_CLASS_COUNT {n_classes}u",
        f"#define CAR_RF_TREE_COUNT {len(roots)}u",
        f"#define CAR_RF_NODE_COUNT {node_count}u", "",
        "/* Class indices map to these labels, in order: */",
    ]
    for index, class_name in enumerate(classes):
        enum_name = "CAR_RF_CLASS_" + "_".join(
            part.upper() for part in __import__("re").split(r"[^A-Za-z0-9]+", class_name) if part
        )
        header.append(f"/* {index}: {class_name} */")
        header.append(f"#define {enum_name} {index}u")
    header += [
        "", "typedef struct {", "    float feature_sum[CAR_RF_FEATURE_COUNT];",
        "    uint32_t windows_seen;", "} car_rf_clip_accumulator_t;", "",
        "/* Add each 1-second feature vector from the clip (11 windows for 6 s at 50% overlap).",
        "   finish averages the 65 features, then returns one diagnosis for the full clip. */",
        "void car_rf_clip_reset(car_rf_clip_accumulator_t *accumulator);",
        "int car_rf_clip_add_window(car_rf_clip_accumulator_t *accumulator,",
        "                           const float features[CAR_RF_FEATURE_COUNT]);",
        "/* Returns the class index, or -1 if no window was added. */",
        "int car_rf_clip_finish(const car_rf_clip_accumulator_t *accumulator,",
        "                      float mean_probabilities[CAR_RF_CLASS_COUNT]);",
        "const char *car_rf_class_name(int class_index);", "",
        "#endif /* CAR_GLOBAL_RF_MODEL_H */", "",
    ]

    def c_array(name: str, ctype: str, values: list, per_line: int = 12) -> list[str]:
        lines = [f"static const {ctype} {name}[{len(values)}] = {{"]
        for start in range(0, len(values), per_line):
            lines.append("    " + ", ".join(str(value) for value in values[start:start + per_line]) + ",")
        lines.append("};")
        return lines

    def c_float(value: float) -> str:
        literal = f"{value:.9g}"
        if "." not in literal and "e" not in literal.lower():
            literal += ".0"
        return literal + "f"

    source = [
        '#include "global_random_forest_model.h"', "", "#include <string.h>", "",
    ]
    source += c_array("node_feature", "int16_t", all_features)
    source += c_array("node_threshold", "float", [c_float(value) for value in all_thresholds])
    source += c_array("node_left", "int32_t", all_left)
    source += c_array("node_right", "int32_t", all_right)
    source += c_array("tree_root", "uint32_t", roots)
    source.append(f"static const uint8_t leaf_prob[{node_count}][{n_classes}] = {{")
    source.extend("    {" + ", ".join(map(str, row)) + "}," for row in all_leaf_probabilities)
    source += [
        "};", "",
        "static void predict_clip_features(const float *features, float *probabilities) {",
        "    for (uint32_t c = 0; c < CAR_RF_CLASS_COUNT; ++c) probabilities[c] = 0.0f;",
        "    for (uint32_t t = 0; t < CAR_RF_TREE_COUNT; ++t) {",
        "        int32_t node = (int32_t)tree_root[t];",
        "        while (node_feature[node] >= 0) {",
        "            uint32_t f = (uint32_t)node_feature[node];",
        "            node = features[f] <= node_threshold[node] ? node_left[node] : node_right[node];",
        "        }",
        "        for (uint32_t c = 0; c < CAR_RF_CLASS_COUNT; ++c)",
        "            probabilities[c] += (float)leaf_prob[node][c] * (1.0f / 255.0f);",
        "    }",
        "    for (uint32_t c = 0; c < CAR_RF_CLASS_COUNT; ++c)",
        "        probabilities[c] *= (1.0f / (float)CAR_RF_TREE_COUNT);",
        "}", "",
        "void car_rf_clip_reset(car_rf_clip_accumulator_t *accumulator) {",
        "    if (accumulator == NULL) return;",
        "    memset(accumulator, 0, sizeof(*accumulator));", "}", "",
        "int car_rf_clip_add_window(car_rf_clip_accumulator_t *accumulator, const float *features) {",
        "    if (accumulator == NULL || features == NULL) return -1;",
        "    for (uint32_t f = 0; f < CAR_RF_FEATURE_COUNT; ++f)",
        "        accumulator->feature_sum[f] += features[f];",
        "    accumulator->windows_seen += 1u;",
        "    return 0;", "}", "",
        "int car_rf_clip_finish(const car_rf_clip_accumulator_t *accumulator, float *mean_probabilities) {",
        "    if (accumulator == NULL || accumulator->windows_seen == 0u) return -1;",
        "    float clip_features[CAR_RF_FEATURE_COUNT];",
        "    for (uint32_t f = 0; f < CAR_RF_FEATURE_COUNT; ++f)",
        "        clip_features[f] = accumulator->feature_sum[f] / (float)accumulator->windows_seen;",
        "    float local_probabilities[CAR_RF_CLASS_COUNT];",
        "    predict_clip_features(clip_features, local_probabilities);",
        "    int best = 0;",
        "    for (uint32_t c = 0; c < CAR_RF_CLASS_COUNT; ++c) {",
        "        float value = local_probabilities[c];",
        "        if (mean_probabilities != NULL) mean_probabilities[c] = value;",
        "        if (c > 0u && value > local_probabilities[best]) best = (int)c;",
        "    }", "    return best;", "}", "",
        "const char *car_rf_class_name(int class_index) {",
        "    static const char *const names[CAR_RF_CLASS_COUNT] = {",
    ]
    source.extend("        " + c_string(name) + "," for name in classes)
    source += [
        "    };",
        "    if (class_index < 0 || (uint32_t)class_index >= CAR_RF_CLASS_COUNT) return \"unknown\";",
        "    return names[class_index];", "}", "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "global_random_forest_model.h").write_text("\n".join(header), encoding="utf-8")
    (output_dir / "global_random_forest_model.c").write_text("\n".join(source), encoding="utf-8")
    return {"nodes": node_count, "trees": len(roots), "classes": classes,
            "max_tree_depth": maximum_depth,
            "exported_leaf_probabilities": "uint8 normalized per leaf"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("car_diagnostics_features/features_65.csv"))
    parser.add_argument("--split", type=Path, default=Path("car_diagnostics_results/recording_split.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/global_random_forest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--clip-sec", type=float, default=6.0)
    args = parser.parse_args()
    if args.trees < 1 or args.max_depth < 1 or args.min_samples_leaf < 1:
        parser.error("trees, max-depth, and min-samples-leaf must be positive")

    frame = pd.read_csv(args.input)
    split = pd.read_csv(args.split)
    features = [name for name in frame.columns if name not in METADATA_COLUMNS]
    if len(features) != 65:
        raise ValueError(f"Expected 65 acoustic features, found {len(features)}")
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    frame["diagnosis"] = diagnosis_labels(frame)
    recordings = aggregate_recordings(frame, features)
    classes = sorted(recordings.diagnosis.unique().tolist())
    if "normal" not in classes or len(classes) < 3:
        raise ValueError("Expected normal plus multiple abnormal fault classes")
    split_frames = {name: split_rows(recordings, split, name)
                    for name in ("train", "validation", "test")}
    if any(data.empty for data in split_frames.values()):
        raise ValueError("Train, validation, and test must all contain recordings")
    split_files = [set(data.file.astype(str).unique()) for data in split_frames.values()]
    if any(split_files[i] & split_files[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("The supplied split contains recording overlap")

    train = split_frames["train"]
    validation = split_frames["validation"]
    test = split_frames["test"]
    tree_candidates = sorted(set([args.trees, args.trees * 2]))
    depth_candidates = sorted(set([max(4, args.max_depth - 2), args.max_depth, args.max_depth + 2]))
    leaf_candidates = sorted(set([1, args.min_samples_leaf]))
    candidates = []
    best_model = None
    best_parameters = None
    best_validation_metrics = None
    best_key = None
    for n_estimators in tree_candidates:
        for max_depth in depth_candidates:
            for min_samples_leaf in leaf_candidates:
                candidate = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf, max_features="sqrt",
                    class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed,
                )
                candidate.fit(train[features], train.diagnosis)
                val_result = recording_predictions(validation, candidate, features)
                val_metrics = score_recordings(val_result, classes)
                candidates.append({
                    "n_estimators": n_estimators, "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "accuracy": val_metrics["accuracy"],
                    "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "macro_f1": val_metrics["macro_f1"],
                })
                key = (val_metrics["macro_f1"], val_metrics["balanced_accuracy"])
                if best_key is None or key > best_key:
                    best_key = key
                    best_model = candidate
                    best_parameters = {"n_estimators": n_estimators, "max_depth": max_depth,
                                       "min_samples_leaf": min_samples_leaf}
                    best_validation_metrics = val_metrics
                print(f"Validation RF: trees={n_estimators}, depth={max_depth}, "
                      f"leaf={min_samples_leaf}, macro-F1={val_metrics['macro_f1']:.3f}", flush=True)

    model = best_model
    selected_parameters = best_parameters
    validation_metrics = best_validation_metrics
    test_results = recording_predictions(test, model, features)
    test_metrics = score_recordings(test_results, classes)
    cm = confusion_matrix(test_results.truth, test_results.prediction, labels=classes)

    # Refit the frozen configuration on train+validation for the deployable artifact;
    # leave test recordings out of the saved model.
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = RandomForestClassifier(
        n_estimators=selected_parameters["n_estimators"],
        max_depth=selected_parameters["max_depth"],
        min_samples_leaf=selected_parameters["min_samples_leaf"],
        max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
        random_state=args.seed,
    ).fit(train_validation[features], train_validation.diagnosis)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    ).to_csv(args.output / "validation_candidates.csv", index=False)
    bundle = {"model": final_model, "feature_names": features, "class_names": classes,
              "normal_label": "normal", "prediction_unit": "6-second recording",
              "window_sec": args.window_sec, "overlap": args.overlap}
    joblib.dump(bundle, args.output / "global_random_forest_normal_plus_faults.joblib", compress=3)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(args.output / "test_confusion_matrix.csv")
    pd.DataFrame({"file": test_results.file, "true_diagnosis": test_results.truth,
                  "predicted_diagnosis": test_results.prediction,
                  "windows_aggregated": test_results.window_count}).to_csv(
                      args.output / "test_recording_predictions.csv", index=False)
    export_metadata = export_c(final_model, args.output / "stm32")
    metadata = {
        "classifier": "RandomForestClassifier", "task": "single-model normal-or-fault-type classification",
        "classes": classes, "normal_behavior": "normal is an explicit output class",
        "feature_count": len(features), "features": features,
        "training_unit": "one training row per recording; each feature is averaged across that recording's overlapping windows",
        "trained_recordings": {name: int(data.file.nunique()) for name, data in split_frames.items()},
        "validation_recording_metrics": validation_metrics,
        "held_out_test_recording_metrics": test_metrics,
        "model_parameters": {**selected_parameters, "max_features": "sqrt",
                             "class_weight": "balanced_subsample", "random_seed": args.seed},
        "parameter_selection": "selected by validation recording macro-F1, balanced accuracy tie-break; test was not used",
        "deployment": {"clip_seconds": args.clip_sec, "window_seconds": args.window_sec,
                       "overlap": args.overlap,
                       "windows_per_clip": int(np.floor((args.clip_sec - args.window_sec) /
                                                         (args.window_sec * (1 - args.overlap)))) + 1,
                       "decision": "average each feature across all windows of the clip, then choose the highest-probability class",
                       "stm32_export": export_metadata,
                       "feature_extraction_note": "C export consumes the same 65 precomputed features per window; audio filtering and feature extraction remain separate."},
        "data_duration_caveat": "Current feature CSV has 2 windows per source recording (starts 0.0 and 0.5 s). The STM32 interface accepts all windows from a 6-second clip, but this test set does not directly validate 6-second recordings.",
        "final_model_fit": "train+validation recordings; held-out test recordings are excluded",
    }
    (args.output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "recording_metrics.json").write_text(json.dumps({
        "validation": validation_metrics, "test": test_metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"validation_recording_metrics": validation_metrics,
                      "test_recording_metrics": test_metrics,
                      "classes": classes, "stm32_export": export_metadata,
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
