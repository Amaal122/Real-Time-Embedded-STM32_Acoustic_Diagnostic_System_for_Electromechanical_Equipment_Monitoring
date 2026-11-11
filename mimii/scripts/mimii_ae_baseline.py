#!/usr/bin/env python3
"""MIMII-style normal-only log-Mel autoencoder baseline using installed scikit-learn."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    # Slaney mel scale, matching librosa's default htk=False behavior.
    f_sp = 200.0 / 3.0
    mel = freq / f_sp
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    log_mask = freq >= min_log_hz
    mel[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mel


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3.0
    freq = mel * f_sp
    min_log_hz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    log_mask = mel >= min_log_mel
    freq[log_mask] = min_log_hz * np.exp(logstep * (mel[log_mask] - min_log_mel))
    return freq


def mel_filterbank(sr: int, n_fft: int = 1024, n_mels: int = 64) -> np.ndarray:
    m_min, m_max = hz_to_mel(np.array([0.0, sr / 2.0]))
    mel_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points.copy())
    fft_freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    fdiff = np.diff(hz_points)
    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for m in range(n_mels):
        lower = -ramps[m] / max(fdiff[m], np.finfo(float).eps)
        upper = ramps[m + 2] / max(fdiff[m + 1], np.finfo(float).eps)
        weights[m] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / np.maximum(hz_points[2:n_mels + 2] - hz_points[:n_mels], np.finfo(float).eps)
    weights *= enorm[:, None]
    return weights


def load_channel_zero(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(path)
    if audio.ndim != 2 or audio.shape[1] < 1:
        raise ValueError(f"Expected multichannel MIMII WAV: {path}")
    if np.issubdtype(audio.dtype, np.integer):
        # librosa scales signed PCM by 2**(bits-1).
        scale = float(2 ** (np.iinfo(audio.dtype).bits - 1))
        signal = audio[:, 0].astype(np.float32) / scale
    else:
        signal = audio[:, 0].astype(np.float32)
    return sr, signal


def logmel_vectors(path: Path, frames: int = 5, max_fft_frames: int | None = None) -> tuple[np.ndarray, int]:
    sr, signal = load_channel_zero(path)
    n_fft, hop, n_mels = 1024, 512, 64
    padded = np.pad(signal, (n_fft // 2, n_fft // 2), mode="constant")
    frame_count = 1 + (len(padded) - n_fft) // hop
    indices = np.arange(n_fft)[None, :] + hop * np.arange(frame_count)[:, None]
    window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    spectrum = np.abs(rfft(padded[indices] * window[None, :], axis=1)) ** 2
    mel = mel_filterbank(sr, n_fft, n_mels).astype(np.float32) @ spectrum.T.astype(np.float32)
    logmel = 10.0 * np.log10(np.maximum(mel, np.finfo(np.float32).eps))
    if frame_count < frames:
        return np.empty((0, n_mels * frames), dtype=np.float32), sr
    vectors = np.concatenate([logmel[:, offset:offset + frame_count - frames + 1].T
                              for offset in range(frames)], axis=1)
    if max_fft_frames is not None and len(vectors) > max_fft_frames:
        positions = np.linspace(0, len(vectors) - 1, max_fft_frames, dtype=int)
        vectors = vectors[positions]
    return vectors.astype(np.float32), sr


def sample_train_vectors(paths: list[Path], frames_per_file: int, seed: int) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    chunks, rates = [], []
    for i, path in enumerate(paths, start=1):
        vectors, sr = logmel_vectors(path)
        rates.append(sr)
        if len(vectors) > frames_per_file:
            chosen = np.sort(rng.choice(len(vectors), size=frames_per_file, replace=False))
            vectors = vectors[chosen]
        chunks.append(vectors)
        if i % 200 == 0 or i == len(paths):
            print(f"Prepared normal training WAVs {i}/{len(paths)}", flush=True)
    return np.vstack(chunks), rates


def score_files(model: MLPRegressor, paths: list[Path]) -> tuple[list[dict], list[int]]:
    rows, rates = [], []
    for i, path in enumerate(paths, start=1):
        vectors, sr = logmel_vectors(path)
        rates.append(sr)
        reconstruction = model.predict(vectors)
        score = float(np.mean(np.mean((vectors - reconstruction) ** 2, axis=1)))
        rows.append({"file": str(path), "anomaly_score": score})
        if i % 100 == 0 or i == len(paths):
            print(f"Scored test WAVs {i}/{len(paths)}", flush=True)
    return rows, rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("0_dB_slider/slider"))
    parser.add_argument("--output", type=Path, default=Path("xgboost_comparison/mimii_ae_baseline"))
    parser.add_argument("--train-ids", nargs="+", default=["id_00", "id_02"])
    parser.add_argument("--test-id", default="id_06")
    parser.add_argument("--frames-per-file", type=int, default=32,
                        help="Subsample log-Mel vectors per normal training WAV to keep this CPU run manageable")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.test_id in args.train_ids:
        parser.error("Test model ID must not be in --train-ids")

    train_paths = sorted(path for model_id in args.train_ids
                         for path in (args.input / model_id / "normal").glob("*.wav"))
    test_paths = []
    labels = []
    for label_name, label_value in (("normal", 0), ("abnormal", 1)):
        paths = sorted((args.input / args.test_id / label_name).glob("*.wav"))
        test_paths.extend(paths)
        labels.extend([label_value] * len(paths))
    if not train_paths or not test_paths or len(set(labels)) != 2:
        raise ValueError("Could not find normal training WAVs and both test classes")

    X_train, rates = sample_train_vectors(train_paths, args.frames_per_file, args.seed)
    if len(set(rates)) != 1:
        raise ValueError(f"Training WAV sample rates differ: {sorted(set(rates))}")
    # Official MIMII baseline architecture: 320 -> 64 -> 64 -> 8 -> 64 -> 64 -> 320.
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64, 8, 64, 64), activation="relu", solver="adam",
        alpha=0.0, batch_size=512, learning_rate_init=0.001, max_iter=args.epochs,
        shuffle=True, random_state=args.seed, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=8, tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, X_train)

    test_rows, test_rates = score_files(model, test_paths)
    if len(set(test_rates)) != 1 or test_rates[0] != rates[0]:
        raise ValueError("Training and test sample rates differ")
    scores = np.array([row["anomaly_score"] for row in test_rows])
    y_true = np.asarray(labels, dtype=np.int8)
    auc = float(roc_auc_score(y_true, scores))

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "mimii_style_autoencoder.joblib")
    predictions = pd.DataFrame(test_rows)
    predictions["true_label"] = np.where(y_true == 1, "abnormal", "normal")
    predictions.to_csv(args.output / "test_scores.csv", index=False)
    results = {
        "method": "MIMII-style normal-only log-Mel dense autoencoder",
        "implementation_note": "Official architecture and feature settings approximated with scikit-learn MLPRegressor because TensorFlow/Keras/librosa are unavailable. Training log-Mel vectors are deterministically subsampled per WAV.",
        "train_ids": args.train_ids, "test_id": args.test_id,
        "train_normal_wavs": len(train_paths), "test_wavs": len(test_paths),
        "test_normal_wavs": int(np.sum(y_true == 0)), "test_abnormal_wavs": int(np.sum(y_true == 1)),
        "train_vectors": int(len(X_train)), "frames_per_normal_wav": args.frames_per_file,
        "sample_rate": int(rates[0]), "channel": 1,
        "logmel": {"n_mels": 64, "n_fft": 1024, "hop_length": 512, "frames_per_vector": 5},
        "hidden_layers": [64, 64, 8, 64, 64], "epochs_requested": args.epochs,
        "epochs_run": int(model.n_iter_), "test_recording_level_roc_auc": auc,
        "test_score_direction": "higher reconstruction MSE means more anomalous",
        "test_id_was_seen_in_prior_experiments": True,
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
