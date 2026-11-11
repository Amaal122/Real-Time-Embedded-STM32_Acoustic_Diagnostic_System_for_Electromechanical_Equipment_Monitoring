#!/usr/bin/env python3
"""Train a small log-Mel CNN for the car dataset's 10 abnormal fault classes."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without optional CNN dependency
    raise SystemExit(
        "PyTorch is required to train this CNN. Install a CPU PyTorch build, then rerun this script."
    ) from exc


def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, low_hz: float, high_hz: float) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(max(0, left), min(center, filters.shape[1])):
            filters[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(max(0, center), min(right, filters.shape[1])):
            filters[m - 1, k] = (right - k) / max(right - center, 1)
    return filters


def read_mono_16k(path: Path, target_length: int) -> np.ndarray:
    sr, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"Expected a mono WAV, got {audio.shape} in {path}")
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    target_sr = 16000
    if sr != target_sr:
        ratio = Fraction(target_sr, int(sr)).limit_denominator()
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    sos = butter(4, [300, 7000], btype="bandpass", fs=target_sr, output="sos")
    audio = sosfiltfilt(sos, audio)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio.astype(np.float32)


def make_logmel(audio: np.ndarray, filters: np.ndarray, n_fft: int = 512,
                frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::hop_length]
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)) ** 2
    mel_power = filters @ spectrum.T
    return np.log(np.maximum(mel_power, 1e-10)).astype(np.float32)


class SmallAudioCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def spec_augment(batch: torch.Tensor) -> torch.Tensor:
    # Small frequency/time masks after standardization. Zero corresponds to training mean.
    out = batch.clone()
    _, _, n_mels, n_frames = out.shape
    for index in range(len(out)):
        if random.random() < 0.8:
            width = random.randint(0, min(8, n_mels // 4))
            start = random.randint(0, max(0, n_mels - width))
            out[index, :, start:start + width, :] = 0
        if random.random() < 0.8:
            width = random.randint(0, min(20, n_frames // 4))
            start = random.randint(0, max(0, n_frames - width))
            out[index, :, :, start:start + width] = 0
    return out


def predict_logits(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(x[start:start + batch_size].to(device)).cpu().numpy())
    return np.concatenate(outputs)


def metric_summary(y: np.ndarray, pred: np.ndarray, class_names: list[str]) -> dict:
    labels = np.arange(len(class_names))
    return {
        "recordings": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/car diagnostics dataset"))
    parser.add_argument("--split", type=Path, default=Path("car_diagnostics_results/recording_split.csv"))
    parser.add_argument("--output", type=Path, default=Path("car_diagnostics_results/cnn"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = pd.read_csv(args.split)
    split = split[split.label == "abnormal"].copy()
    split_map = {row.file: row.split for row in split.itertuples(index=False)}
    class_names = sorted(split.fault_type.unique())
    encoder = LabelEncoder().fit(class_names)
    target_length = int(round(1.7 * 16000))
    filters = mel_filterbank(16000, 512, 64, 300, 7000)

    audio_specs, records = [], []
    files = sorted(split.file.unique())
    for index, file_key in enumerate(files, start=1):
        path = args.dataset / Path(file_key)
        audio = read_mono_16k(path, target_length)
        spec = make_logmel(audio, filters)
        row = split.loc[split.file == file_key].iloc[0]
        audio_specs.append(spec)
        records.append({"file": file_key, "fault_type": row.fault_type,
                        "condition": row.condition, "split": split_map[file_key]})
        if index % 200 == 0 or index == len(files):
            print(f"Prepared {index}/{len(files)} abnormal WAV spectrograms", flush=True)
    x_all = np.stack(audio_specs).astype(np.float32)
    records = pd.DataFrame(records)
    train_idx = np.flatnonzero(records.split.to_numpy() == "train")
    val_idx = np.flatnonzero(records.split.to_numpy() == "validation")
    test_idx = np.flatnonzero(records.split.to_numpy() == "test")
    y_all = encoder.transform(records.fault_type)

    # Normalize each Mel band using training recordings only.
    means = x_all[train_idx].mean(axis=(0, 2), keepdims=True)
    stds = x_all[train_idx].std(axis=(0, 2), keepdims=True)
    stds = np.maximum(stds, 1e-4)
    x_all = (x_all - means) / stds
    x = torch.from_numpy(x_all[:, None, :, :].astype(np.float32))
    y = torch.from_numpy(y_all.astype(np.int64))
    train_loader = DataLoader(TensorDataset(x[train_idx], y[train_idx]), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    counts = np.bincount(y_all[train_idx], minlength=len(class_names)).astype(np.float32)
    class_weights = len(train_idx) / (len(class_names) * np.maximum(counts, 1))
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    model = SmallAudioCNN(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_f1, best_state, best_epoch, wait = -1.0, None, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = spec_augment(batch_x).to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = predict_logits(model, x[val_idx], device)
        val_pred = val_logits.argmax(axis=1)
        val_result = metric_summary(y_all[val_idx], val_pred, class_names)
        val_f1 = val_result["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": val_f1,
                        "validation_balanced_accuracy": val_result["balanced_accuracy"]})
        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: loss={np.mean(losses):.4f}, "
                  f"val_macro_f1={val_f1:.4f}, val_bal_acc={val_result['balanced_accuracy']:.4f}", flush=True)
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation macro-F1={best_f1:.4f}", flush=True)
            break

    model.load_state_dict(best_state)
    test_logits = predict_logits(model, x[test_idx], device)
    test_pred = test_logits.argmax(axis=1)
    test_result = metric_summary(y_all[test_idx], test_pred, class_names)
    test_records = records.iloc[test_idx].copy()
    test_records["true_class"] = test_records.fault_type
    test_records["predicted_class"] = encoder.inverse_transform(test_pred)
    test_records.to_csv(args.output / "test_predictions.csv", index=False)
    torch.save({
        "state_dict": best_state,
        "class_names": class_names,
        "mean_per_mel_band": means.reshape(-1).tolist(),
        "std_per_mel_band": stds.reshape(-1).tolist(),
        "target_sample_rate": 16000,
        "target_duration_sec": 1.7,
        "n_mels": 64,
        "n_fft": 512,
        "frame_length": 400,
        "hop_length": 160,
        "architecture": "SmallAudioCNN: 16-32-64 conv channels, global average pooling",
    }, args.output / "fault_type_cnn.pt")
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    confusion = pd.DataFrame(test_result.pop("confusion_matrix"), index=class_names, columns=class_names)
    confusion.to_csv(args.output / "test_confusion_matrix.csv")
    result = {
        "model": "CNN on 64-band log-Mel spectrograms",
        "task": "10-class fault type classification, abnormal recordings only",
        "selected_epoch_by_validation_macro_f1": best_epoch,
        "best_validation_macro_f1": float(best_f1),
        "train_recordings": int(len(train_idx)),
        "validation_recordings": int(len(val_idx)),
        "test_recordings": int(len(test_idx)),
        "test_metrics": test_result,
        "framework": f"PyTorch {torch.__version__}",
        "device": str(device),
        "note": "CNN epoch selected on validation; test was measured for this run but has also been used in previous project model comparisons.",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
