#!/usr/bin/env python3
"""Record a microphone clip, then run the Random Forest fault specialists."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-sec", type=float, default=30.0,
                        help="recording duration (default: 30 seconds)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="microphone sample rate; must be above 14 kHz")
    parser.add_argument("--device", help="optional microphone device index or name")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("car_diagnostics_results/fault_specialists"),
                        help="directory containing the Random Forest specialist models")
    parser.add_argument("--threshold-strategy", choices=["max_f1", "target_recall", "lcd_table"],
                        default="max_f1", help="threshold profile for the specialist models")
    parser.add_argument("--recording", type=Path,
                        default=Path("car_diagnostics_results/microphone_capture.wav"),
                        help="where to save the microphone WAV")
    parser.add_argument("--predictions", type=Path,
                        help="optional per-window CSV output path")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        parser.error(
            "sounddevice is required for microphone input; install it with "
            "the same Python interpreter: python -m pip install sounddevice"
        )

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.sample_rate <= 14000:
        parser.error("--sample-rate must be greater than 14 kHz for the 7 kHz filter")

    args.recording.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording from the selected microphone for {args.duration_sec:g} seconds.")
    print("Play the car sound from the other PC now; keep the speakers near this microphone.")
    try:
        audio = sd.rec(
            int(round(args.duration_sec * args.sample_rate)),
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=args.device,
        )
        sd.wait()
    except Exception as exc:
        parser.error(f"microphone recording failed: {exc}")

    wavfile.write(args.recording, args.sample_rate, audio[:, 0])
    print(f"Recorded WAV: {args.recording.resolve()}")

    script_dir = Path(__file__).resolve().parent
    predictor = script_dir / "predict_wav_fault_specialists.py"
    command = [
        sys.executable, str(predictor),
        "--input", str(args.recording.resolve()),
        "--model-dir", str(args.model_dir.resolve()),
        "--threshold-strategy", args.threshold_strategy,
    ]
    if args.predictions:
        command.extend(["--output", str(args.predictions.resolve())])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
