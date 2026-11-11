# Car acoustic diagnostics

This branch contains the car diagnostic training and evaluation scripts, deployable model artifacts, aggregate metrics, and presentation files. Raw recordings, extracted feature tables, recording splits, and per-recording prediction exports are excluded.

See `results/README_FR.md` for the experiment notes and `results/` for summary metrics and saved models. To rerun the pipeline, provide the excluded data locally and install the dependencies listed by the scripts.

The saved global Random Forest fault-type classifier is `results/fault_model_search/selected_fault_classifier.joblib`. It predicts the fault class among abnormal car recordings; normal-vs-abnormal detection uses the separate XGBoost model.
