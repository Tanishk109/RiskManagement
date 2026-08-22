# Metrics artifacts

Generated validation and held-out metrics are written here by the ML scripts. `baseline_validation.json`, `catboost_validation.json`, `experiments.csv`, `threshold_analysis.json`, and `threshold_grid.parquet` contain actual validation-only development evidence; they are not product-facing final metrics. Threshold cost outputs are labelled **ESTIMATED BUSINESS COST UNDER USER-SUPPLIED ASSUMPTIONS**.

`final_test_metrics.json` is the only product source of truth after final evaluation. `final_test_predictions.csv` contains protected row-level outputs and is ignored by git.
