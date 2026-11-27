# Predicting O-Level `final_test` Scores

A machine learning pipeline that predicts a student's `final_test` score from their background, habits and school details. It uses a **random forest tuned with RandomizedSearchCV**, saves the trained model, and can predict scores for new students.

The analysis behind the choices in the pipeline is in [eda.ipynb](eda.ipynb). The pipeline itself runs from [main.py](main.py).

---

## Project structure

```
regression_olvl/
├── data/
│   └── data.csv              # raw dataset (15,900 students)
├── src/
│   ├── config.yaml           # all settings: cleaning, features, split, model
│   ├── data_preparation.py   # DataPreparation: load -> clean -> engineer features -> split
│   └── model_training.py     # ModelTraining: tune -> evaluate -> save outputs
├── eda.ipynb                 # exploratory data analysis and model comparison
├── main.py                   # trains the model, or predicts new students with --predict
├── outputs/                  # created by main.py: saved model, metrics, predictions, run record
├── requirements.txt
└── README.md
```

---

## How to run

1. Install the dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline from the project root:

   ```bash
   python main.py
   ```

   To use a different config file:

   ```bash
   python main.py --config path/to/config.yaml
   ```

3. Predict `final_test` for new students with the saved model (run step 2 first):

   ```bash
   python main.py --predict path/to/new_students.csv
   ```

   The CSV needs the same columns as `data/data.csv`; `final_test` can be left out. Labels are fixed and missing `attendance_rate` is filled with the saved training median, exactly as in training. Results go to `outputs/predictions.csv`.

Training prints the train/test metrics next to a mean baseline, the score on each cross-validation fold, and the top 10 features. It saves these files to `outputs/`:

| File | What it holds |
|---|---|
| `model.joblib` | The fitted pipeline (preprocessor + random forest), used by `--predict` |
| `model_record.json` | Chosen model, best settings, CV mean and std, test and baseline metrics, fill values, package versions |
| `metrics.csv` | Train, test and baseline metrics |
| `cv_folds.csv` | Score on each cross-validation fold, plus mean and std |
| `test_predictions.csv` | Actual vs predicted score for every test student |
| `feature_importances.csv` | How much the forest relies on each feature |
| `config_used.yaml` | A copy of the config used for the run |

To change a setting (for example the test size, the number of search iterations or the forest settings to search), edit [src/config.yaml](src/config.yaml). You don't need to change the code.

---

## Pipeline

```
data.csv ──> Load ──> Clean ──> Engineer features ──> Train/test split (80/20)
                                                              │
  Save to outputs/ <── Evaluate (vs baseline) <── RandomizedSearchCV (5-fold CV) on
                                               [Preprocessor ──> Random forest] pipeline
```

### 1. Data cleaning

| Issue | Fix |
|---|---|
| `tuition` uses both `Y`/`N` and `Yes`/`No` | Map to `Yes`/`No` |
| `CCA` has mixed casing and missing values | Capitalise; fill missing with `None` |
| `age` has typos `5` and `6` | Map to `15` and `16` |
| Duplicate rows (exact copies of another row) | Drop the copies (139 rows) |
| Missing `final_test` (the target) | Drop the row |
| Invalid ages (≤ 0) | Drop the row |
| Missing `attendance_rate` (~5%) | Keep the row; fill with the **training-set median** (95) after the split, so the test set doesn't leak in |

This removes 638 rows (4%), leaving 15,262.

### Input checks

The script stops early, with a message naming the problem, if:

- the config file is missing, isn't valid YAML, or is missing a required key (listed in `REQUIRED_CONFIG_KEYS` in [main.py](main.py))
- a config value doesn't make sense, for example `test_size` outside 0–1 or `cv` below 2
- the data file is missing, is empty, or is missing a required column, including the target (`REQUIRED_COLUMNS` in [src/data_preparation.py](src/data_preparation.py))
- a feature named in the config doesn't exist, for example a typo
- no rows are left after cleaning, or there are fewer training rows than CV folds

Rows with a badly written `sleep_time` or `wake_time` are dropped with a warning instead of crashing the run. A warning is also logged if cleaning removes more than 20% of the rows.

### 2. Feature engineering

| Feature | How it is built | Why |
|---|---|---|
| `sleep_hours` | `wake_time` − `sleep_time`, handling bedtimes before midnight | Sleep duration is more useful than the raw clock times |
| `class_size` | `n_male` + `n_female` | A stronger predictor than either column alone |
| `study_band` | `hours_per_week` grouped into 0–4, 5–10, 11+ hrs | Scores peak at 5–10 hrs, which a straight line can't capture |
| `low_attendance` | `attendance_rate` ≤ 80 | Scores drop sharply below this threshold |
| `enough_sleep` | `sleep_hours` ≥ 7 | Scores drop sharply below this threshold |

The columns that showed little or no relationship with `final_test` in the EDA are dropped: `gender`, `age`, `mode_of_transport`, `bag_color` and `student_id`.

### 3. Preprocessing

Done inside a scikit-learn `Pipeline`, so it is fitted on the training folds only:

- **Numerical** (`class_size`, `attendance_rate`, `sleep_hours`, `low_attendance`, `enough_sleep`): `StandardScaler`. Missing `attendance_rate` is filled with the training median just before this step.
- **Nominal** (`CCA`, `learning_style`, `tuition`, `direct_admission`, `study_band`, `number_of_siblings`): `OneHotEncoder(drop="first")`

Scaling doesn't change a random forest's results (trees split on thresholds, not distances). It is kept so the same preprocessor works for every model compared in the notebook.

### 4. Model

- **Random forest** (`RandomForestRegressor`). The search picks from `n_estimators` {100, 200, 300}, `max_depth` {None, 10, 15, 20}, `min_samples_leaf` {1, 2, 5, 10} and `max_features` {"sqrt", 0.5, 1.0}.
- **Baseline**: a `DummyRegressor` that always predicts the training mean, so the model's scores have something to be compared with.
- **RandomizedSearchCV**: 20 candidates × 5-fold cross-validation, scored on R², run on the **training set only**. The test set is used once, for the final score.

---

## Results

Best parameters: `n_estimators = 200`, `max_depth = 10`, `min_samples_leaf = 2`, `max_features = 1.0`.

| Metric | Train | Test | Baseline (mean) test |
|---|---|---|---|
| MAE | 3.421 | 3.624 | 11.636 |
| RMSE | 4.978 | 5.174 | 13.948 |
| R² | 0.873 | 0.862 | 0.000 |

On unseen students the model predicts `final_test` within about **3.6 marks** on average and explains about **86%** of the variation in scores. The baseline is off by 11.6 marks. The test score is only slightly below the training score (RMSE gap 0.2 marks), so the model is not overfitting much.

**Cross-validation (training set, R²):** 0.844, 0.853, 0.846, 0.849, 0.859. Mean 0.850, std 0.005, so the result is stable across folds.

### Most important features

| Feature | Importance |
|---|---|
| `class_size` | 0.398 |
| `number_of_siblings_2` | 0.137 |
| `study_band_5-10 hrs` | 0.129 |
| `number_of_siblings_1` | 0.113 |
| `learning_style_Visual` | 0.075 |
| `attendance_rate` | 0.062 |
| `direct_admission_Yes` | 0.033 |
| `tuition_Yes` | 0.026 |

Importance shows how much the forest relies on a feature, not whether it raises or lowers the score. For direction, see the Lasso coefficients in [eda.ipynb](eda.ipynb): for example, two siblings is about 11 marks lower and 5–10 study hours about 11 marks higher.

---

## Why a random forest?

The EDA notebook compares a mean baseline, Linear Regression, Ridge and Lasso (each untuned, tuned with GridSearchCV and tuned with RandomizedSearchCV), and a random forest (untuned and tuned with RandomizedSearchCV), all on the same test set.

| Model | Test RMSE | Test R² |
|---|---|---|
| **Random forest + RandomizedSearchCV** | **5.174** | **0.862** |
| Random forest (default) | 5.603 | 0.839 |
| Best linear model (Lasso + RandomizedSearchCV) | 8.575 | 0.622 |
| Baseline (predict the mean) | 13.948 | 0.000 |

The random forest was chosen because:

- **It is by far the most accurate.** It cuts the average error by about 40% compared with the best linear model.
- **It is stable.** Cross-validation R² stays between 0.84 and 0.86 across folds, and train and test scores are close.
- **It fits the patterns in the EDA.** Scores have thresholds and peaks (for example the 5–10 study-hour peak) that a straight line can only approximate.

The trade-off is that a forest has no single coefficient per feature, so it is harder to explain than Lasso. The notebook keeps the Lasso coefficients as an easy-to-read summary of each factor's direction and size.
