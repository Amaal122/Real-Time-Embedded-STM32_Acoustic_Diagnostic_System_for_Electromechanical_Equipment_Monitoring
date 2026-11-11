# 0 dB slider feature pipeline

Run from the project root:

```powershell
python scripts/extract_0db_features.py --input 0_dB_slider --output features_0dB
```

Dependencies: `numpy`, `pandas`, `scipy`, `PyWavelets` (`pywt`), and `scikit-learn`.

To audit the training and validation WAVs for format issues or obvious cancellation in the selected channel pair:

```powershell
python scripts/audit_0db_inputs.py --input 0_dB_slider --output xgboost_comparison/input_audit --subjects id_00 id_02 id_04 --channels 3 5
```

Each run randomly chooses one distinct pair from the eight channels and applies that same pair to every recording, so rows across the dataset use a consistent sensor pair. The pair is EGC averaged, Butterworth bandpass filtered (4th order, 300–7000 Hz), split into 1-second windows with 50% overlap, then analyzed with a level-4 `db4` wavelet packet. Use `--seed 42` to reproduce the random pair, `--pair 2 7` to force a pair, and `--window-sec` / `--overlap` to change windowing.

Outputs:

- `features_65.csv`: class and recording metadata plus the 65 features for every window.
- `feature_selection_comparison.csv`: per-feature JSD and one-way ANOVA scores and ranks, computed across `normal` and `abnormal` windows. This compares the two ranking methods; it does not choose a final feature subset or train/evaluate a classifier.
- `run_metadata.json`: settings and selected channels for the run.

The default 1-second window is configurable. Ensure the WAV sample rate is above 14 kHz for the 7 kHz filter cutoff. Feature-selection scores are a first comparison; for a final model evaluation, split by subject/recording before selection to avoid leakage between overlapping windows.

## XGBoost comparison

After extracting the feature CSV, run the held-out MIMII model-ID comparison from the project root:

```powershell
python scripts/compare_xgboost_selection.py --input features_0dB/features_65.csv --output xgboost_comparison
```

This uses leave-one-model-ID-out evaluation. JSD and ANOVA rankings are recalculated on each training fold, and XGBoost is compared with the top 10 and top 20 features from each method plus a 65-feature baseline. Results are written to `fold_results.csv`, `summary.csv`, `selected_features_by_fold.json`, and `evaluation_metadata.json`. Public MIMII 1.0 has four available slider model IDs, so this is a small exploratory evaluation.

For the earlier fixed split across model IDs, run:

```powershell
python scripts/train_val_test_xgboost.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --output xgboost_comparison/train_val_test --train-subjects id_00 id_02 --validation-subject id_04 --test-subject id_06 --top-k 10
```

This selects ANOVA features on training rows, uses validation data to choose a probability threshold, refits the final model on train+validation, then evaluates once on the held-out test model ID. Earlier exploratory CV included all IDs, so this is a retrospective holdout rather than a pristine independent test. The older `train_final_xgboost.py` fits on all rows and should only be used when an independent evaluation set is not needed.

To train/evaluate the same split with JSD top 10 instead of ANOVA, run:

```powershell
python scripts/train_val_test_xgboost.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --output xgboost_comparison/train_val_test/jsd_top10 --train-subjects id_00 id_02 --validation-subject id_04 --test-subject id_06 --top-k 10 --selection-method JSD
```

This stores a separate model and metrics in `xgboost_comparison/train_val_test/jsd_top10`. Its test metrics are retrospective because `id_06` has already appeared in earlier exploratory analyses.

To compare model settings without using the test subject, run:

```powershell
python scripts/tune_xgboost_train_validation.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --output xgboost_comparison/train_val_test/tuning_feature_counts --train-subjects id_00 id_02 --validation-subject id_04 --top-k 5 10 15 20 30 40 50 60
```

The previous 10/20-feature comparison favored all 65 features, depth 4, and no class weighting on validation (ROC AUC 0.693, balanced accuracy 0.641). This is a candidate selected on one validation model ID, so it may overfit that ID; the test ID was excluded from tuning. Each tuning run saves its candidate model under that run's output directory.

For a wider feature-count sweep, set `--top-k 5 10 15 20 30 40 50 60`. In the current run, JSD top 60 was best on window-level validation (ROC AUC 0.710, balanced accuracy 0.654); ANOVA top 60 scored 0.698 / 0.649, and all 65 features scored 0.693 / 0.641. These differences are small and based on one validation model ID.

Because the target is a WAV recording and its overlapping windows are correlated, the recording-level candidate averages each recording's window probabilities. Refit and evaluate this choice on training/validation only with:

```powershell
python scripts/recording_level_xgboost.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --output xgboost_comparison/recording_level --train-subjects id_00 id_02 --validation-subject id_04 --top-k 60
```

On validation model ID `id_04`, the recording-level metrics were ROC AUC 0.745 and balanced accuracy 0.705 at the validation-selected threshold. Recall was 0.882 and precision 0.384, so it catches more abnormal recordings but also produces false alarms. This is validation on one model ID, not an independent performance estimate.

The frozen cross-model recording-level model was evaluated once on all 623 WAV recordings for `id_06` using the validation threshold. It scored ROC AUC 0.693, balanced accuracy 0.662, abnormal recall 0.820, and abnormal precision 0.216. See `xgboost_comparison/recording_level/test_evaluation/recording_metrics.json` and `recording_predictions.csv`. This test is retrospective because `id_06` was viewed in earlier experiments.

After freezing that candidate, its one-time test evaluation can be run with:

```powershell
python scripts/evaluate_xgboost_test.py --input features_0dB/features_65.csv --model-dir xgboost_comparison/train_val_test/tuned_model --test-subject id_06
```

The resulting `metrics.json` and per-window `predictions.csv` are saved under the model's `test_evaluation` folder. Do not use these test results to tune the same candidate.

## XGBoost per known MIMII model ID

For a known-product task, train a separate model for each slider model ID and split WAV files within that ID. All overlapping windows from one WAV stay in the same split:

```powershell
python scripts/train_per_model_id_xgboost.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --output xgboost_comparison/per_model_id --model-ids id_00 id_02 id_04 id_06 --top-k 60
```

This gave a macro ROC AUC of 0.997 and macro balanced accuracy of 0.969 over the four 15% WAV test splits. These are exploratory because earlier work inspected the same dataset; they estimate performance for known IDs, not transfer to a new product model.

For the same file-level split and XGBoost settings with ANOVA top 60, run the same command with `--output xgboost_comparison/per_model_id_anova --selection-method ANOVA`. The macro ROC AUC was 0.997 and macro balanced accuracy 0.972, effectively tied with JSD on this small, previously inspected dataset.

For a more stable selector comparison, use five-fold cross-validation grouped by WAV within every model ID:

```powershell
python scripts/cv_per_model_id_xgboost.py --input features_0dB/features_65.csv --output xgboost_comparison/per_model_id_cv --model-ids id_00 id_02 id_04 id_06 --top-k 60 --folds 5 --seed 42
```

This fixed-threshold CV gave macro ROC AUC 0.9983 for ANOVA and 0.9982 for JSD; macro balanced accuracy was 0.9654 and 0.9627. Treat the difference as a tie. The source data has been used in earlier experiments, so these are stability estimates rather than a pristine final test.

After comparing selectors, fit deployable JSD models on all WAVs for each known model ID. The decision threshold for each model is set from its grouped out-of-fold recording predictions:

```powershell
python scripts/fit_final_per_model_id_xgboost.py --input features_0dB/features_65.csv --pipeline-metadata features_0dB/run_metadata.json --oof xgboost_comparison/per_model_id_cv/recording_oof_predictions.csv --output xgboost_comparison/final_per_model_id --model-ids id_00 id_02 id_04 id_06 --top-k 60 --seed 42
```

These final models use every labelled WAV, so their CV metrics are the performance estimates; do not evaluate them on those same WAVs as a fresh test.

## MIMII-style normal-only autoencoder check

Run the cross-model-ID comparison with:

```powershell
python scripts/mimii_ae_baseline.py --input 0_dB_slider/slider --output xgboost_comparison/mimii_ae_baseline/train_plus_validation --train-ids id_00 id_02 id_04 --test-id id_06 --frames-per-file 32 --epochs 50
```

This uses the official MIMII baseline's log-Mel settings and dense autoencoder layer sizes, but substitutes scikit-learn for unavailable TensorFlow/Keras/librosa and samples 32 training vectors per normal WAV. It is therefore a lightweight baseline approximation, not a bit-for-bit reproduction. The cross-ID test AUC for this run was 0.493.

To predict a new multichannel WAV with the saved model:

```powershell
python scripts/predict_wav_xgboost.py --input path/to/new_recording.wav --model-id id_06
```

Select the known MIMII model ID with `--model-id`; the script loads it from `xgboost_comparison/final_per_model_id`. Or pass `--model-dir` to load another saved bundle. The script uses the selected model's stored channel pair and preprocessing settings, writes per-window probabilities to a CSV beside the input WAV, and prints a recording summary based on the mean abnormal probability. The input must have the required channels and a sample rate above the stored filter cutoff.
