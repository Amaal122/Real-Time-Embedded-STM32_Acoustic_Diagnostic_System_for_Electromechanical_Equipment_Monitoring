# STM32 car signal-processing module

`car_signal_processing.c/.h` is the C-side feature pipeline for the global car Random Forest. It expects signed PCM16 audio already sampled at 16 kHz; microphone/PDM capture and any sample-rate conversion happen before this API.

The call processes one six-second buffer. It applies the same 4th-order 300–7000 Hz Butterworth bandpass as the training extractor, using forward/backward filtering with odd-edge padding. It then computes eleven 1-second windows at 50% overlap. For each window it calculates the same ordered 65 temporal, spectral, and `db4` level-4 WPT features. The model accumulator averages features across all windows and returns one of the eleven classes, including `normal`.

## CubeIDE integration

1. Add `car_signal_processing.c` and `global_random_forest_model.c` to `Core/Src`; add both headers to `Core/Inc`.
2. Initialize the board's FMC SDRAM before using the signal module. Reserve a `car_signal_workspace_t` in external SDRAM through a linker section; the workspace uses about 690 KiB and must not be placed on the stack or in internal SRAM.
3. Configure the audio path to produce exactly 96,000 PCM16 samples at 16 kHz for each six-second capture. Keep the sample buffer in external SDRAM as well.
4. Call `car_signal_workspace_init()` once SDRAM is available, then call `car_signal_classify_pcm16_6s(buffer, 96000, &workspace, probabilities)`. Convert the returned class with `car_rf_class_name()`.

The Python reference pipeline is in `cars/scripts/extract_car_features.py` and `cars/scripts/extract_0db_features.py`. Preserve the exact 65-feature order from the metadata. JSD ranking in `train_car_models.py` is for training-time feature selection; this global Random Forest consumes all 65 features and does not run JSD on the board.

This repository does not contain the user's complete CubeIDE project or microphone driver. The C module is standalone and still needs to be built and checked against the target project's audio capture and SDRAM configuration before flashing.
