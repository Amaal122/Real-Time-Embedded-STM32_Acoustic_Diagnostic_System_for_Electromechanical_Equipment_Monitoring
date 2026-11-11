#!/usr/bin/env python3
"""Extract 65 window features from a random two-channel EGC pair in 0 dB data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from scipy.stats import entropy, kurtosis, skew
from sklearn.feature_selection import f_classif


def feature_names() -> list[str]:
    return (
        ["t_mean", "t_std", "t_max", "t_min", "t_rms", "t_kurtosis", "t_skewness",
         "t_crest", "t_energy", "t_zcr", "t_mean_abs", "t_peak2peak",
         "s_centroid", "s_bandwidth", "s_rolloff85", "s_flatness", "s_peak_ratio"]
        + [f"w{i:02d}_rms_rel" for i in range(16)]
        + [f"w{i:02d}_kurt" for i in range(16)]
        + [f"w{i:02d}_entropy" for i in range(16)]
    )


def extract_window(x: np.ndarray, sr: int, wavelet: str) -> list[float]:
    eps = np.finfo(float).eps
    x = np.asarray(x, dtype=float)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)))
    centered = x - np.mean(x)

    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total_power = float(np.sum(spectrum))
    if total_power <= eps:
        centroid = bandwidth = rolloff = flatness = peak_ratio = 0.0
    else:
        centroid = float(np.sum(freqs * spectrum) / total_power)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_power))
        rolloff = float(freqs[min(np.searchsorted(np.cumsum(spectrum), 0.85 * total_power), len(freqs) - 1)])
        flatness = float(np.exp(np.mean(np.log(spectrum + eps))) / (np.mean(spectrum) + eps))
        peak_ratio = float(np.max(spectrum) / total_power)

    packet = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=4)
    leaves = packet.get_level(4, order="freq")
    energies = np.array([np.sum(np.asarray(node.data, dtype=float) ** 2) for node in leaves])
    total_energy = float(np.sum(energies))
    rel_rms = np.sqrt(energies / total_energy) if total_energy > eps else np.zeros(16)
    wp_kurt = [float(kurtosis(node.data, fisher=True, bias=False)) if len(node.data) > 3 else 0.0 for node in leaves]
    wp_entropy = []
    for node in leaves:
        coefficients = np.asarray(node.data, dtype=float)
        coefficient_energy = coefficients * coefficients
        energy_sum = float(np.sum(coefficient_energy))
        if energy_sum <= eps:
            wp_entropy.append(0.0)
        else:
            coefficient_probabilities = coefficient_energy / energy_sum
            wp_entropy.append(float(entropy(coefficient_probabilities, base=2)))

    temporal = [
        float(np.mean(x)), float(np.std(x)), float(np.max(x)), float(np.min(x)), rms,
        float(kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else 0.0,
        float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        float(np.max(abs_x) / (rms + eps)), float(np.sum(x * x)),
        float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))),
        float(np.mean(abs_x)), float(np.ptp(x)),
    ]
    return temporal + [centroid, bandwidth, rolloff, flatness, peak_ratio] + rel_rms.tolist() + wp_kurt + wp_entropy


def jsd_score(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Histogram Jensen-Shannon divergence between the two class distributions."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("0_dB_slider"))
    parser.add_argument("--output", type=Path, default=Path("features_0dB"))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5, help="Fractional window overlap, from 0 to <1")
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed to reproduce the random pair")
    parser.add_argument("--pair", nargs=2, type=int, metavar=("CH1", "CH2"), help="Force 1-based channel pair")
    parser.add_argument("--filter-order", type=int, default=4)
    args = parser.parse_args()
    if args.window_sec <= 0 or not 0 <= args.overlap < 1:
        parser.error("--window-sec must be positive and --overlap must be in [0, 1)")

    files = sorted(args.input.rglob("*.wav"))
    if not files:
        parser.error(f"No WAV files found under {args.input}")
    first_sr, first_audio = wavfile.read(files[0])
    if first_audio.ndim != 2 or first_audio.shape[1] < 2:
        parser.error(f"Expected multichannel WAV; got {first_audio.shape} in {files[0]}")
    channels = first_audio.shape[1]
    if args.pair:
        chosen = tuple(ch - 1 for ch in args.pair)
        if chosen[0] == chosen[1] or any(ch < 0 or ch >= channels for ch in chosen):
            parser.error(f"Channel pair must be two distinct 1..{channels} channel numbers")
        chosen = tuple(sorted(chosen))
    else:
        rng = np.random.default_rng(args.seed)
        chosen = tuple(sorted(rng.choice(channels, size=2, replace=False).tolist()))

    rows: list[dict] = []
    names = feature_names()
    skipped = []
    for file in files:
        sr, audio = wavfile.read(file)
        if sr != first_sr or audio.ndim != 2 or audio.shape[1] <= max(chosen):
            skipped.append(str(file))
            continue
        # Scale integer PCM to a consistent floating-point range, then EGC-average the chosen channels.
        if np.issubdtype(audio.dtype, np.integer):
            scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
            audio = audio.astype(np.float64) / scale
        else:
            audio = audio.astype(np.float64)
        signal = (audio[:, chosen[0]] + audio[:, chosen[1]]) / 2.0
        nyquist = sr / 2.0
        if nyquist <= 7000:
            raise ValueError(f"Sample rate {sr} Hz in {file} cannot support a 7 kHz cutoff")
        sos = butter(args.filter_order, [300, 7000], btype="bandpass", fs=sr, output="sos")
        try:
            filtered = sosfiltfilt(sos, signal)
        except ValueError as exc:
            raise ValueError(f"Audio too short for zero-phase Butterworth filtering: {file}") from exc
        win = int(round(args.window_sec * sr))
        hop = max(1, int(round(win * (1.0 - args.overlap))))
        label = file.parent.name.lower()
        if label not in {"normal", "abnormal"}:
            raise ValueError(f"Expected class folder named normal/abnormal, got {file.parent}")
        subject = file.parent.parent.name
        for start in range(0, len(filtered) - win + 1, hop):
            feats = extract_window(filtered[start:start + win], sr, args.wavelet)
            row = dict(zip(names, feats))
            row.update(label=label, subject=subject, file=str(file.relative_to(args.input)),
                       window_start_sec=start / sr, sample_rate=sr,
                       channel_1=chosen[0] + 1, channel_2=chosen[1] + 1)
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete audio windows were extracted")
    metadata_cols = ["label", "subject", "file", "window_start_sec", "sample_rate", "channel_1", "channel_2"]
    frame[metadata_cols + names].to_csv(args.output / "features_65.csv", index=False)

    classes = sorted(frame.label.unique())
    if len(classes) == 2:
        X = frame[names].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = frame.label
        f_values, p_values = f_classif(X, y)
        f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0)
        jsd_values = [jsd_score(X.loc[y == classes[0], c].to_numpy(), X.loc[y == classes[1], c].to_numpy()) for c in names]
        ranking = pd.DataFrame({"feature": names, "jsd_score": jsd_values,
                                "anova_f": f_values, "anova_p": np.nan_to_num(p_values, nan=1.0)})
        ranking["jsd_rank"] = ranking.jsd_score.rank(ascending=False, method="min").astype(int)
        ranking["anova_rank"] = ranking.anova_f.rank(ascending=False, method="min").astype(int)
        ranking.sort_values("jsd_rank").to_csv(args.output / "feature_selection_comparison.csv", index=False)
    else:
        print(f"Feature ranking skipped: expected both classes; found {classes}")

    (args.output / "run_metadata.json").write_text(json.dumps({
        "input": str(args.input), "wavelet": args.wavelet, "wavelet_packet_level": 4,
        "filter": {"type": "Butterworth bandpass", "order": args.filter_order, "low_hz": 300, "high_hz": 7000},
        "window_sec": args.window_sec, "overlap": args.overlap,
        "random_seed": args.seed, "selected_channels_1_based": [chosen[0] + 1, chosen[1] + 1],
        "source_files": len(files), "windows": len(frame), "skipped_files": skipped,
        "feature_count": len(names), "classes": classes,
    }, indent=2), encoding="utf-8")
    print(f"Random channel pair: {chosen[0] + 1} & {chosen[1] + 1}")
    print(f"Extracted {len(frame)} windows with {len(names)} features -> {args.output}")


if __name__ == "__main__":
    main()
