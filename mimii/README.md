# MIMII slider anomaly detection

This branch contains the 0 dB slider feature pipeline, XGBoost and Random Forest comparisons, deployable per-ID XGBoost models, evaluation summaries, and the presentation. Source recordings and extracted window-level feature matrices are excluded.

The global Random Forest vs XGBoost comparison uses a single 80/20 split grouped by WAV recording. All four model IDs occur in both splits; JSD top 60 is selected from training rows only. See `results/random_forest_comparison/mixed_ids_grouped_split/recording_metrics.csv` for recording-level metrics.

Run scripts from this `mimii` directory after providing the dataset and installing dependencies described in `README_feature_pipeline.md`.
