#!/usr/bin/env python3
"""Run the six calibrated car-fault specialist models on one WAV recording."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

from extract_0db_features import extract_window, feature_names
from extract_car_features import pcm_to_float


FAULT_DATASET_NAMES = {
    "worn_out_brakes": "worn_out_brakes",
    "bad_ignition": "bad_ignition",
    "dead_battery": "dead_battery",
    "low_oil": "low_oil",
    "power_steering": "power_steering",
    "serpentine_belt": "serpentine_belt",
}

# Values transcribed from the user's reference table. These are a manual profile,
# not thresholds calibrated for the RandomForest specialists in this project.
LCD_TABLE_THRESHOLDS = {
    "worn_out_brakes": 0.90,
    "bad_ignition": 0.75,
    "low_oil": 0.45,
    "power_steering": 0.40,
    "serpentine_belt": 0.35,
}


def extract_recording_features(
    input_path: Path,
    target_sr: int = 16000,
    window_sec: float = 1.0,
    overlap: float = 0.5,
    wavelet: str = "db4",
    filter_order: int = 4,
) -> tuple[pd.DataFrame, int]:
    if target_sr <= 14000:
        raise ValueError("target sample rate must be greater than 14 kHz for the 7 kHz filter")
    if window_sec <= 0 or not 0 <= overlap < 1:
        raise ValueError("window-sec must be positive and overlap must be in [0, 1)")

    source_sr, audio = wavfile.read(input_path)
    if source_sr <= 14000:
        raise ValueError(f"source sample rate {source_sr} Hz cannot support the 7 kHz passband")
    signal = pcm_to_float(audio)
    if source_sr != target_sr:
        ratio = Fraction(target_sr, int(source_sr)).limit_denominator()
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)

    sos = butter(filter_order, [300, 7000], btype="bandpass", fs=target_sr, output="sos")
    filtered = sosfiltfilt(sos, signal)
    window_samples = int(round(window_sec * target_sr))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    starts = list(range(0, len(filtered) - window_samples + 1, hop_samples))
    if not starts:
        raise ValueError(f"recording is shorter than the {window_sec:g} second feature window")

    names = feature_names()
    rows = []
    for start in starts:
        values = extract_window(filtered[start:start + window_samples], target_sr, wavelet)
        rows.append(dict(zip(names, values)))
    return pd.DataFrame(rows, columns=names), int(source_sr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="path to the input WAV")
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("car_diagnostics_results/fault_specialists"),
        help="directory containing detector_*.joblib and threshold_calibration/recommended_thresholds.json",
    )
    parser.add_argument(
        "--threshold-strategy", choices=["max_f1", "target_recall", "lcd_table"], default="max_f1",
        help="threshold profile; lcd_table applies the values from the supplied reference table",
    )
    parser.add_argument("--output", type=Path, help="CSV output path; defaults beside the input WAV")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--filter-order", type=int, default=4)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input WAV does not exist: {args.input}")
    if args.input.suffix.lower() != ".wav":
        parser.error("input must be a .wav file")

    calibration_path = args.model_dir / "threshold_calibration" / "recommended_thresholds.json"
    if not calibration_path.is_file():
        parser.error(f"calibrated threshold file not found: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    profiles = calibration["threshold_profiles_for_full_data_models"]

    windows, source_sr = extract_recording_features(
        args.input, args.target_sr, args.window_sec, args.overlap,
        args.wavelet, args.filter_order,
    )
    rows = []
    detected = []
    for fault in FAULT_DATASET_NAMES:
        model_path = args.model_dir / f"detector_{fault}.joblib"
        if not model_path.is_file():
            parser.error(f"specialist model not found: {model_path}")
        bundle = joblib.load(model_path)
        expected_features = bundle["features"]
        if list(windows.columns) != list(expected_features):
            missing = sorted(set(expected_features) - set(windows.columns))
            extra = sorted(set(windows.columns) - set(expected_features))
            raise ValueError(
                f"feature schema mismatch for {fault}; missing={missing}, extra={extra}"
            )
        profile_key = f"{args.threshold_strategy}_threshold_median"
        if args.threshold_strategy == "lcd_table":
            threshold = LCD_TABLE_THRESHOLDS.get(
                fault, float(profiles[fault]["max_f1_threshold_median"])
            )
        else:
            threshold = float(profiles[fault][profile_key])
        model = bundle["model"]
        positive_col = int(np.flatnonzero(model.classes_ == 1)[0])
        probabilities = model.predict_proba(windows[expected_features])[:, positive_col]
        mean_probability = float(np.mean(probabilities))
        is_detected = mean_probability >= threshold
        if is_detected:
            detected.append(fault)
        for i, probability in enumerate(probabilities):
            rows.append({
                "window_index": i,
                "window_start_sec": i * args.window_sec * (1.0 - args.overlap),
                "fault": fault,
                "fault_folder_name": FAULT_DATASET_NAMES[fault],
                "window_probability": float(probability),
                "recording_mean_probability": mean_probability,
                "threshold": threshold,
                "detected_for_recording": bool(is_detected),
                "threshold_strategy": args.threshold_strategy,
            })

    output_path = args.output or args.input.with_name(f"{args.input.stem}_fault_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"WAV: {args.input}")
    print(f"Sample rate: {source_sr} Hz -> {args.target_sr} Hz; windows: {len(windows)}")
    print(f"Threshold strategy: {args.threshold_strategy}")
    print("Recording-level results (mean of window probabilities):")
    for fault in FAULT_DATASET_NAMES:
        fault_rows = [row for row in rows if row["fault"] == fault]
        mean_probability = fault_rows[0]["recording_mean_probability"]
        threshold = fault_rows[0]["threshold"]
        status = "DETECTED" if fault in detected else "not detected"
        print(f"  {FAULT_DATASET_NAMES[fault]}: {status} (score={mean_probability:.4f}, threshold={threshold:.4f})")
    if detected:
        print("Detected component(s): " + ", ".join(FAULT_DATASET_NAMES[fault] for fault in detected))
    else:
        print("No specialist threshold was crossed; this does not prove that the recording is normal.")
    print(f"Per-window results: {output_path.resolve()}")


if __name__ == "__main__":
    main()
