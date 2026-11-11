#!/usr/bin/env python3
"""Predict normal/abnormal labels for windows in a new multichannel WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from xgboost import XGBClassifier

from extract_0db_features import extract_window, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="New multichannel WAV recording")
    parser.add_argument("--model-root", type=Path, default=Path("xgboost_comparison/final_per_model_id"))
    parser.add_argument("--model-id", choices=["id_00", "id_02", "id_04", "id_06"], default=None,
                        help="Known MIMII slide-rail product model ID; selects its dedicated classifier")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Explicit model folder; overrides --model-id for older single-model bundles")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV; defaults beside input")
    parser.add_argument("--threshold", type=float, default=None, help="Abnormal probability threshold (default: model metadata)")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input WAV does not exist: {args.input}")

    if args.model_dir is not None:
        model_dir = args.model_dir
    elif args.model_id is not None:
        model_dir = args.model_root / args.model_id
    else:
        parser.error("Set --model-id for the per-model-ID classifiers, or provide --model-dir explicitly")

    bundle = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    selected = json.loads((model_dir / bundle["selected_features_file"]).read_text(encoding="utf-8"))
    pipeline = bundle["preprocessing_from_feature_extraction_run"]
    channels = [int(ch) - 1 for ch in pipeline["selected_channels_1_based"]]
    filt = pipeline["filter"]
    window_sec = float(bundle["preprocessing_from_feature_extraction_run"]["window_sec"])
    overlap = float(bundle["preprocessing_from_feature_extraction_run"]["overlap"])
    wavelet = str(pipeline["wavelet"])
    threshold = float(bundle["prediction_threshold"] if args.threshold is None else args.threshold)
    if not 0.0 <= threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    sr, audio = wavfile.read(args.input)
    if audio.ndim != 2 or audio.shape[1] <= max(channels):
        raise ValueError(f"Expected a multichannel WAV with at least {max(channels) + 1} channels; got {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    signal = (audio[:, channels[0]] + audio[:, channels[1]]) / 2.0

    low_hz, high_hz = float(filt["low_hz"]), float(filt["high_hz"])
    if sr / 2.0 <= high_hz:
        raise ValueError(f"Sample rate {sr} Hz cannot support the {high_hz:g} Hz filter cutoff")
    sos = butter(int(filt["order"]), [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError as exc:
        raise ValueError("Audio is too short for the configured zero-phase filter") from exc

    win = int(round(window_sec * sr))
    hop = max(1, int(round(win * (1.0 - overlap))))
    names = feature_names()
    rows = []
    for start in range(0, len(filtered) - win + 1, hop):
        values = extract_window(filtered[start:start + win], sr, wavelet)
        rows.append(dict(zip(names, values)))
    if not rows:
        raise ValueError(f"Recording is shorter than one {window_sec:g}-second window")

    features = pd.DataFrame(rows)[selected]
    model = XGBClassifier()
    model.load_model(model_dir / bundle["model_file"])
    probabilities = model.predict_proba(features)[:, 1]
    predictions = np.where(probabilities >= threshold, "abnormal", "normal")
    result = pd.DataFrame({
        "window_start_sec": [start / sr for start in range(0, len(filtered) - win + 1, hop)],
        "probability_normal": 1.0 - probabilities,
        "probability_abnormal": probabilities,
        "predicted_label": predictions,
    })
    output = args.output or args.input.with_name(f"{args.input.stem}_predictions.csv")
    result.to_csv(output, index=False)
    mean_abnormal = float(np.mean(probabilities))
    overall_label = "abnormal" if mean_abnormal >= threshold else "normal"
    print(f"Windows: {len(result)}")
    print(f"Mean abnormal probability: {mean_abnormal:.4f}")
    print(f"Recording summary at threshold {threshold:g}: {overall_label}")
    print(f"Per-window predictions: {output}")


if __name__ == "__main__":
    main()
