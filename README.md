# LLM-Assisted Optuna Hyperparameter Optimization

This project compares two different Optuna approaches for hyperparameter optimization (tuning) on ​​a Scikit-learn `RandomForestRegressor` model:

1. **LLM-Assisted Method:** Using the Google Gemini API, a dataset-specific, logical, and narrowed-down search space is generated based on dataset statistics (number of rows/columns).
2. **Traditional Method:** Standard, widely accepted, and very broad hyperparameter ranges found in the literature are used.

The project demonstrates how the LLM-assisted search achieves similar success (R²) scores using much lighter, shallower models (lower `max_depth` and `n_estimators`) that are better suited for production environments.

## Project Comparison Results

The following results were obtained from Optuna optimization runs (10 trials each) performed on the California Housing regression dataset (~20,640 rows):

* **LLM-Assisted Model Score (R²):** 0.6582
  * *Parameters Found:* `{'max_depth': 13, 'n_estimators': 88}`

* **Traditional Model Score (R²):** 0.6587
  * *Parameters Found:* `{'max_depth': 49, 'n_estimators': 121}`

### Analysis of Results
The traditional method drastically increased model complexity (`max_depth: 49`) for a microscopic score gain of only **0.0005**. This indicates a tendency for the model to memorize the data (overfitting) rather than learning from it. The LLM-assisted method, on the other hand, achieved the same level of success by setting reasonable initial boundaries for Optuna, resulting in a model that was much shallower (depth of 13) and comprised fewer trees (88 trees).
