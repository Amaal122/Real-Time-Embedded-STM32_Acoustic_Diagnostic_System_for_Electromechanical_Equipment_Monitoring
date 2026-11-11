#!/usr/bin/env python3
"""Extract MIMII-style 65 features from the mono car-diagnostics WAV dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

from extract_0db_features import extract_window, feature_names


STATE_NAMES = {
    "braking state": "braking",
    "idle state": "idle",
    "startup state": "startup",
}


def describe_file(path: Path, root: Path) -> tuple[str, str, str, str]:
    parts = path.relative_to(root).parts
    if len(parts) < 3:
        raise ValueError(f"Expected condition/class/file directory structure, got: {path}")
    condition_dir = parts[0].lower()
    if condition_dir not in STATE_NAMES:
        raise ValueError(f"Unknown operating-state folder: {parts[0]}")
    condition = STATE_NAMES[condition_dir]
    class_parts = parts[1:-1]
    raw_class = "__".join(class_parts).strip().lower()
    # Include condition so similarly named classes from different states remain distinct.
    fault_type = f"{condition}__{raw_class}"
    leaf = class_parts[-1].strip().lower()
    label = "normal" if leaf.startswith("normal_") else "abnormal"
    return condition, raw_class, fault_type, label


def pcm_to_float(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected mono or multichannel PCM WAV, got shape {audio.shape}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        return audio.astype(np.float64) / scale
    return audio.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/car diagnostics dataset"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_features"))
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--filter-order", type=int, default=4)
    args = parser.parse_args()
    if args.target_sr <= 14000:
        parser.error("--target-sr must exceed 14 kHz for the 7 kHz filter cutoff")
    if args.window_sec <= 0 or not 0 <= args.overlap < 1:
        parser.error("--window-sec must be positive and --overlap must be in [0, 1)")

    files = sorted(args.input.rglob("*.wav"))
    if not files:
        parser.error(f"No WAV files found under {args.input}")

    rows: list[dict] = []
    names = feature_names()
    source_rates: Counter[int] = Counter()
    source_channels: Counter[int] = Counter()
    skipped: list[str] = []
    sos = butter(args.filter_order, [300, 7000], btype="bandpass", fs=args.target_sr, output="sos")
    win = int(round(args.window_sec * args.target_sr))
    hop = max(1, int(round(win * (1.0 - args.overlap))))

    for index, file in enumerate(files, start=1):
        try:
            sr, audio = wavfile.read(file)
            if sr <= 14000:
                raise ValueError(f"Source sample rate {sr} Hz cannot support a 7 kHz passband")
            source_rates[int(sr)] += 1
            source_channels[int(audio.shape[1] if audio.ndim == 2 else 1)] += 1
            signal = pcm_to_float(audio)
            if sr != args.target_sr:
                ratio = Fraction(args.target_sr, int(sr)).limit_denominator()
                signal = resample_poly(signal, ratio.numerator, ratio.denominator)
            filtered = sosfiltfilt(sos, signal)
            condition, raw_class, fault_type, label = describe_file(file, args.input)
            file_key = file.relative_to(args.input).as_posix()
            starts = list(range(0, len(filtered) - win + 1, hop))
            if not starts:
                raise ValueError(f"Clip is shorter than the {args.window_sec:g} second window")
            for start in starts:
                values = extract_window(filtered[start:start + win], args.target_sr, args.wavelet)
                row = dict(zip(names, values))
                row.update(
                    label=label,
                    condition=condition,
                    fault_type=fault_type,
                    class_in_state=raw_class,
                    file=file_key,
                    window_start_sec=start / args.target_sr,
                    sample_rate=args.target_sr,
                )
                rows.append(row)
        except Exception as exc:
            skipped.append(f"{file.relative_to(args.input).as_posix()}: {exc}")
        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} WAV files", flush=True)

    if skipped:
        print(f"Skipped {len(skipped)} file(s); see run_metadata.json", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete feature windows were extracted")
    metadata_cols = ["label", "condition", "fault_type", "class_in_state", "file", "window_start_sec", "sample_rate"]
    args.output.mkdir(parents=True, exist_ok=True)
    frame[metadata_cols + names].to_csv(args.output / "features_65.csv", index=False)

    recordings = frame.drop_duplicates("file")
    recording_counts = recordings.groupby(["condition", "fault_type", "label"], sort=True).size()
    inventory = recording_counts.rename("recordings").reset_index()
    inventory.to_csv(args.output / "recording_inventory.csv", index=False)
    metadata = {
        "input": str(args.input),
        "source_files_found": len(files),
        "source_sample_rates_hz": dict(sorted(source_rates.items())),
        "source_channel_counts": dict(sorted(source_channels.items())),
        "target_sample_rate_hz": args.target_sr,
        "resampling": "scipy.signal.resample_poly when source rate differs from target",
        "filter": {"type": "Butterworth bandpass", "order": args.filter_order, "low_hz": 300, "high_hz": 7000},
        "mono_handling": "mono used directly; multichannel input downmixed by arithmetic mean",
        "window_sec": args.window_sec,
        "overlap": args.overlap,
        "wavelet": args.wavelet,
        "wavelet_packet_level": 4,
        "feature_count": len(names),
        "recordings_extracted": int(recordings.file.nunique()),
        "windows_extracted": len(frame),
        "skipped_files": skipped,
        "classes": inventory.to_dict(orient="records"),
        "note": "Feature ranking is not computed here; it must be fit using training recordings only.",
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Extracted {len(frame)} windows / {recordings.file.nunique()} recordings with {len(names)} features -> {args.output}")


if __name__ == "__main__":
    main()
