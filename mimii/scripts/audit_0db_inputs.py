#!/usr/bin/env python3
"""Audit labels and channel-pair signal quality on training/validation subjects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("0_dB_slider"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/input_audit"))
    parser.add_argument("--subjects", nargs="+", default=["id_00", "id_02", "id_04"])
    parser.add_argument("--channels", nargs=2, type=int, default=[3, 5], metavar=("CH1", "CH2"), help="1-based channel pair")
    args = parser.parse_args()
    if args.channels[0] == args.channels[1] or min(args.channels) < 1:
        parser.error("--channels must be two different positive 1-based channel numbers")

    records = []
    files = []
    for subject in args.subjects:
        for label in ("normal", "abnormal"):
            folder = args.input / "slider" / subject / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing expected label folder: {folder}")
            files.extend((path, subject, label) for path in sorted(folder.glob("*.wav")))
    if not files:
        raise FileNotFoundError("No WAV files found for requested subjects")

    channel_a, channel_b = (ch - 1 for ch in args.channels)
    expected_sample_rate = None
    expected_channels = None
    for index, (path, subject, label) in enumerate(files, start=1):
        sr, audio = wavfile.read(path)
        if audio.ndim != 2:
            raise ValueError(f"Not a multichannel recording: {path}, shape={audio.shape}")
        if audio.shape[1] <= max(channel_a, channel_b):
            raise ValueError(f"Missing selected channel(s) in {path}: shape={audio.shape}")
        if expected_sample_rate is None:
            expected_sample_rate, expected_channels = sr, audio.shape[1]
        centered = audio.astype(np.float64)
        if np.issubdtype(audio.dtype, np.integer):
            max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            clipped_fraction = float(np.mean(np.abs(audio.astype(np.int64)) >= max_value))
        else:
            clipped_fraction = float(np.mean(np.abs(centered) >= 1.0))
        a = centered[:, channel_a]
        b = centered[:, channel_b]
        a = a - np.mean(a)
        b = b - np.mean(b)
        rms_a = float(np.sqrt(np.mean(a * a)))
        rms_b = float(np.sqrt(np.mean(b * b)))
        egc = (a + b) / 2.0
        egc_rms = float(np.sqrt(np.mean(egc * egc)))
        denom = (rms_a + rms_b) / 2.0
        cancellation_ratio = egc_rms / denom if denom > 0 else 0.0
        corr = float(np.corrcoef(a, b)[0, 1]) if rms_a > 0 and rms_b > 0 else np.nan
        records.append({
            "subject": subject, "label": label, "file": str(path.relative_to(args.input)),
            "sample_rate": int(sr), "channels": int(audio.shape[1]),
            "duration_sec": len(audio) / sr,
            "channel_a_rms": rms_a, "channel_b_rms": rms_b,
            "pair_correlation": corr, "egc_cancellation_ratio": cancellation_ratio,
            "clipped_fraction": clipped_fraction,
            "silent_pair": bool(rms_a < 1e-10 or rms_b < 1e-10),
        })
        if index % 200 == 0 or index == len(files):
            print(f"Audited {index}/{len(files)} WAV files", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(records)
    detail.to_csv(args.output / "file_audit.csv", index=False)
    summary = detail.groupby(["subject", "label"], as_index=False).agg(
        recordings=("file", "count"), median_duration_sec=("duration_sec", "median"),
        sample_rates=("sample_rate", lambda x: ",".join(map(str, sorted(set(x))))),
        channel_counts=("channels", lambda x: ",".join(map(str, sorted(set(x))))),
        median_pair_correlation=("pair_correlation", "median"),
        anti_correlated_fraction=("pair_correlation", lambda x: float(np.mean(x < -0.5))),
        median_egc_cancellation_ratio=("egc_cancellation_ratio", "median"),
        severe_cancellation_fraction=("egc_cancellation_ratio", lambda x: float(np.mean(x < 0.5))),
        silent_pair_fraction=("silent_pair", "mean"), max_clipped_fraction=("clipped_fraction", "max"),
    )
    summary.to_csv(args.output / "summary_by_subject_class.csv", index=False)
    print(f"Subjects: {args.subjects}; selected channels: {args.channels}")
    print(f"Sample rates: {sorted(detail.sample_rate.unique().tolist())}; channel counts: {sorted(detail.channels.unique().tolist())}")
    print(f"WAV files: {len(detail)}; sample-rate consistent: {detail.sample_rate.nunique() == 1}; channel-count consistent: {detail.channels.nunique() == 1}")
    print(f"Silent selected pair: {int(detail.silent_pair.sum())}; files with any clipping: {int((detail.clipped_fraction > 0).sum())}")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
