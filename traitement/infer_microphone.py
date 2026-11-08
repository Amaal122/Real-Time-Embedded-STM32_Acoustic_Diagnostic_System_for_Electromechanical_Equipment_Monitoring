#!/usr/bin/env python3
"""Inference XGBoost from the default microphone over 30 one-second windows."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from extract_features import window_features


FS = 16000
WINDOW = FS
FILTER = signal.butter(2, [300.0, 7000.0], btype="bandpass", fs=FS, output="sos")


def record(seconds):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice n'est pas installé : pip install sounddevice")
    samples = int(seconds * FS)
    print(f"Enregistrement de {seconds:.0f} secondes avec le microphone par défaut...")
    audio = sd.rec(samples, samplerate=FS, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pump_xgb_all.json")
    parser.add_argument("--meta", default=None)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--global-ratio", type=float, default=0.5,
                        help="fraction minimale de fenêtres anormales pour déclarer le son anormal")
    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta) if args.meta else Path(str(model_path) + ".meta")
    if not model_path.is_file():
        parser.error(f"modèle introuvable : {model_path}")
    if not meta_path.is_file():
        parser.error(f"métadonnées introuvables : {meta_path}")

    with meta_path.open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    if metadata.get("variant") != "all":
        parser.error("ce script attend un modèle variant='all' avec 65 features")

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    raw = record(args.seconds)
    n_windows = len(raw) // WINDOW
    if n_windows == 0:
        parser.error("enregistrement trop court")
    raw = raw[:n_windows * WINDOW]
    filtered = signal.sosfilt(FILTER, raw.astype(np.float64)).reshape(n_windows, WINDOW)
    groups = ["temporal", "spectral", "wpt"]
    features = np.asarray([window_features(w, groups) for w in filtered], dtype=np.float32)
    expected = int(metadata["feature_count"])
    if features.shape[1] != expected:
        parser.error(f"{features.shape[1]} features calculées, {expected} attendues")

    probabilities = model.predict_proba(features)[:, 1]
    threshold = float(metadata["threshold"])
    abnormal = probabilities >= threshold
    ratio = float(abnormal.mean())
    decision = "ANORMAL" if ratio >= args.global_ratio else "NORMAL"

    print("\nRésultat par fenêtre :")
    for i, (probability, is_abnormal) in enumerate(zip(probabilities, abnormal), 1):
        label = "anormal" if is_abnormal else "normal"
        print(f"  fenêtre {i:02d}: {label} (probabilité={probability:.3f})")
    print("\nDécision globale :", decision)
    print(f"Fenêtres anormales : {int(abnormal.sum())}/{n_windows} ({ratio:.1%})")
    print(f"Seuil modèle : {threshold:.3f} ; seuil global : {args.global_ratio:.1%}")


if __name__ == "__main__":
    main()
