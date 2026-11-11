# Car audio signal-processing pipeline

## Preprocessing and windows

`extract_car_features.py` reads mono WAV files, converts integer PCM to floating point, resamples to 16 kHz, applies a 4th-order Butterworth bandpass from 300 to 7000 Hz with `scipy.signal.sosfiltfilt`, then uses 1-second windows with 50% overlap. For a 6-second clip at those settings, that yields 11 windows.

## The 65 features and WPT

`extract_0db_features.py` provides the shared `extract_window` implementation and exact feature order: 12 temporal features, 5 spectral features, and 48 wavelet-packet features. The WPT uses `db4`, symmetric extension, level 4, and frequency-ordered leaves. The 48 WPT values are 16 relative RMS values, 16 kurtosis values, and 16 coefficient entropies.

## JSD

`train_car_models.py` computes Jensen-Shannon divergence feature rankings during training/model comparison. JSD is a training-time feature-selection statistic; it does not process audio and is not run on the STM32 for each clip. The current normal-plus-fault global Random Forest uses all 65 features, so its inference path does not use JSD-selected features.

## STM32

The Python extractor defines the reference preprocessing/features, but it is not firmware code. The exported `global_random_forest_model.c` only classifies 65 feature values. The STM32 must implement equivalent resampling/filtering, windowing, WPT, and feature calculations in C before calling the classifier. Preserve this exact feature order.
