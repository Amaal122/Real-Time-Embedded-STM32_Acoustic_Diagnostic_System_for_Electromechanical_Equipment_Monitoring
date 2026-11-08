# Sonic AI

**Real-Time Embedded STM32 Acoustic Diagnostic System for Electromechanical Equipment Monitoring**

Sonic AI listens to a running engine (or, more broadly, any electromechanical machine) and identifies mechanical faults from sound alone — entirely on-device, in real time, with **no cloud, no phone, and no network connection**. The full pipeline (audio capture → feature extraction → classification → decision → display) runs on an STM32 microcontroller.

> Built as an engineering internship project at **ACTIA Group**, in partnership with **Synergy Lab (SUPCOM)** and the **University of Carthage**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware](#hardware)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Fault Categories](#fault-categories)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Mechanical faults almost always announce themselves acoustically before they become critical — a worn belt squeals, a failing brake grinds, a misfire produces an irregular signature. Sonic AI turns this qualitative, experience-based diagnostic skill into a repeatable, always-available, automatic check performed **on the edge**.

**Why edge computing?**
- ⚡ **Low latency** — diagnosis is immediate
- 🔒 **Privacy** — audio never leaves the device
- 💸 **Low cost** — no infrastructure or connectivity required
- 📶 **Reliability** — works in the noisy, network-poor environments where machines actually operate

## Key Features

- 🎙️ Real-time audio acquisition from a MEMS microphone (PDM → PCM conversion on-device)
- 🧠 65-dimensional hand-crafted feature vector (wavelet-packet, temporal, spectral)
- 🌲 Compact gradient-boosted tree classifier (XGBoost), ~7.86 KB compiled footprint
- 🔄 Full deployment path: Python training → ONNX export → in-IDE validation → compiled embedded C
- 🖥️ Dual result reporting: on-board color display **and** a desktop dashboard over UART
- 🔧 Validated on two hardware platforms (STM32F429-DISCO → STM32F469I-DISCO)

## System Architecture

```
MEMS Microphone
      │
      ▼
 PDM → PCM Conversion
      │
      ▼
 Feature Extraction (65 features: wavelet-packet, temporal, spectral)
      │
      ▼
 Gradient-Boosted Model (XGBoost, on-device inference)
      │
      ▼
 Decision Logic (calibrated per-class thresholds)
      │
      ▼
 Output: On-board LCD (color-coded) + UART → Desktop Dashboard
```

Audio is captured continuously via DMA into a one-second rolling window; once a window is complete, it is converted, featurized, classified, and the verdict is displayed — while the next window is already being captured.

## Hardware

| Component | Detail |
|---|---|
| Development platform (initial) | STM32F429-DISCO (Cortex-M4, FPU) |
| Development platform (final) | STM32F469I-DISCO (Cortex-M4, on-board MEMS mic, DSI display, external SDRAM) |
| Microphone | On-board MEMS (PDM output) |
| Display | On-board LCD (color-coded verdict) |
| Communication | UART to companion desktop dashboard |

## Machine Learning Pipeline

**Dataset:** Public *Car Diagnostics Dataset* (Kaggle) — engine audio across braking, idle, and start-up states, healthy and faulty.

**Model exploration:** A large Audio Spectrogram Transformer (AST) teacher (F1 ≈ 0.96, not deployable) was used to guide knowledge distillation into six compact neural students (CNN2D, DSCNN, TCResNet8, CNN1D_WPT, MobileNetV1Lite, GRU). A gradient-boosted tree ensemble (XGBoost) trained directly on hand-crafted features ultimately outperformed every deployable neural candidate while being cheaper and more stable to train.

**Chosen model:** XGBoost on a 65-feature vector (48 wavelet-packet + 12 temporal + 5 spectral features), trained with cross-validation, exported to ONNX, validated, and compiled into firmware via STM32Cube.AI.

## Results

| Metric | Value |
|---|---|
| Classes | 6 (1 healthy + 5 faults) |
| Feature vector dimension | 65 |
| Model | XGBoost (~21,708 tree nodes) |
| Compiled footprint | ~7.86 KB flash |
| Macro-averaged F1 (cross-validated) | 0.820 (out-of-fold: 0.818) |
| Window length | 1 second |

## Repository Structure

```
sonic-ai/
├── firmware/           # STM32CubeIDE project (audio acquisition, feature extraction, inference, display)
├── ml/                 # Python training pipeline (data prep, augmentation, model exploration)
│   ├── data/            # Dataset processing scripts
│   ├── models/           # Teacher, student, and XGBoost training scripts
│   └── export/            # ONNX export & validation
├── dashboard/          # Desktop companion app (serial dashboard)
├── docs/                # Report, diagrams, figures
└── README.md
```

*(Adjust to match your actual folder layout before pushing.)*

## Getting Started

### Firmware
1. Open `firmware/` in **STM32CubeIDE**.
2. Flash to an STM32F469I-DISCO (or STM32F429-DISCO with an external microphone).
3. Power on — the board starts classifying one-second audio windows continuously.

### Desktop Dashboard
```bash
cd dashboard
pip install -r requirements.txt
python dashboard.py --port <YOUR_SERIAL_PORT> --baud 115200
```

### Retraining the Model
```bash
cd ml
pip install -r requirements.txt
python train_xgboost.py
python export_onnx.py
```

## Fault Categories

| Index | Class | Nature |
|---|---|---|
| 0 | Brakes | Braking-system noise |
| 1 | Ignition | Irregular combustion / misfire |
| 2 | Normal | Healthy reference condition |
| 3 | Oil | Lubrication / oil-pressure anomaly |
| 4 | Power steering | Power-steering system strain |
| 5 | Serpentine belt | Accessory-drive belt fault |

## Lessons Learned

- **Instrument the middle of the pipeline, not just the output.** A frozen or default prediction can stem from the microphone, DMA, interrupt routing, features, or the model — inspecting the intermediate PCM buffer is what localizes the fault.
- **Know when to abandon an approach.** Manual peripheral configuration was scrapped in favor of the board's own BSP audio driver once it became a dead end.
- **Match the model to the constraints**, not to the state of the art — the most powerful model isn't useful if it doesn't fit the device.
- **Treat manual firmware additions as first-class artifacts.** IDE configuration regeneration can silently wipe hand-written driver code and interrupt wiring — document and back it up.

## Future Work

- Rebuild the on-board display on the manufacturer's command-mode driver (fixes NT35510 panel tearing)
- Temporal smoothing across windows to reduce momentary misclassifications
- On-device logging for trend analysis / predictive maintenance
- Broader fault coverage across more machines and fault types
- Further model compression / distillation to close the gap with the teacher model

## Acknowledgements

This project was carried out at **Synergy Lab**, hosted at **SUPCOM** (École Supérieure des Communications de Tunis, University of Carthage), in partnership with the **ACTIA Group**.

## License

*Add your chosen license here (e.g., MIT, Apache 2.0).*
