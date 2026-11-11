# Global car Random Forest

This is a single 11-class model. It returns `normal` for normal audio, or one of the 10 fault labels for abnormal audio. The existing XGBoost binary detector remains a separate normal/anomaly model.

The trainer first averages each acoustic feature across all windows in a recording, then trains one recording-level classifier. On the STM32, add each 1-second window feature vector to the accumulator and call `car_rf_clip_finish` once after the full 6-second clip. With 1-second windows and 50% overlap, that is 11 windows. The C model consumes 65 engineered features per window; audio filtering and feature extraction are still required before this classifier.

Current test metrics are accuracy 0.597, balanced accuracy 0.529, and macro-F1 0.538 over 278 held-out recordings. The current feature CSV has only two windows per source WAV, so these results do not directly validate 6-second clips. Confirm the longer-clip pipeline on representative 6-second recordings before relying on board predictions.

Files: `global_random_forest_normal_plus_faults.joblib` is the Python bundle; `cars/embedded/global_random_forest_model.c` and `.h` are the STM32 inference export. `test_recording_predictions.csv` and all source recordings/features were intentionally omitted.
