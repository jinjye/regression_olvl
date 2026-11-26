# Predicting O-Level `final_test` Scores

A machine learning pipeline that predicts a student's `final_test` score from their background, habits and school details. It uses **Lasso regression tuned with RandomizedSearchCV**.

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
│   └── model_training.py     # ModelTraining: tune -> evaluate -> coefficients
├── eda.ipynb                 # exploratory data analysis and model comparison
├── main.py                   # runs the full pipeline
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

The script prints the train/test metrics and the top 10 features. It does not write any files, such as a saved model.

To change a setting (for example the test size, the number of search iterations or the alpha range), edit [src/config.yaml](src/config.yaml). You don't need to change the code.

---

## Pipeline

```
data.csv ──> Load ──> Clean ──> Engineer features ──> Train/test split (80/20)
                                                              │
      Print results <── Evaluate <── RandomizedSearchCV (5-fold CV) on
                                      [Preprocessor ──> Lasso] pipeline
```

### 1. Data cleaning

| Issue | Fix |
|---|---|
| `tuition` uses both `Y`/`N` and `Yes`/`No` | Map to `Yes`/`No` |
| `CCA` has mixed casing and missing values | Capitalise; fill missing with `None` |
| `age` has typos `5` and `6` | Map to `15` and `16` |
| Missing `final_test` or `attendance_rate` | Drop the row |
| Invalid ages (≤ 0) | Drop the row |

This removes 1,257 rows, leaving 14,643.

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

- **Numerical** (`class_size`, `attendance_rate`, `sleep_hours`, `low_attendance`, `enough_sleep`): `StandardScaler`
- **Nominal** (`CCA`, `learning_style`, `tuition`, `direct_admission`, `study_band`, `number_of_siblings`): `OneHotEncoder(drop="first")`

### 4. Model

- **Lasso regression**, with `alpha` sampled from a log-uniform range of 0.0001–1000 and `fit_intercept` in {True, False}.
- **RandomizedSearchCV**: 30 candidates × 5-fold cross-validation, scored on R², run on the **training set only**. The test set is used once, for the final score.

---

## Results

Best parameters: `alpha ≈ 0.0136`, `fit_intercept = True` (CV R² 0.613).

| Metric | Train | Test | Gap (test − train) |
|---|---|---|---|
| MAE | 6.908 | 6.851 | −0.057 |
| RMSE | 8.652 | 8.610 | −0.043 |
| R² | 0.614 | 0.625 | +0.011 |

On unseen students the model predicts `final_test` within about **6.9 marks** on average and explains about **63%** of the variation in scores. The train and test scores are almost the same, so the model is not overfitting.

### Most important features

| Feature | Coefficient | Effect on score |
|---|---|---|
| `number_of_siblings_2` | −10.70 | Two siblings (vs none): about 10.7 marks lower |
| `study_band_5-10 hrs` | +10.10 | Studying 5–10 hrs/week (vs 0–4): about 10.1 marks higher |
| `number_of_siblings_1` | −6.43 | One sibling (vs none): lower |
| `CCA_None` | +6.01 | No CCA: higher |
| `study_band_11+ hrs` | +4.95 | Studying 11+ hrs/week: higher, but less than 5–10 hrs |
| `class_size` | −4.77 | Larger classes: lower |
| `learning_style_Visual` | +4.74 | Visual learners: higher |
| `tuition_Yes` | +3.97 | Taking tuition: higher |

---

## Why Lasso + RandomizedSearchCV?

The EDA notebook compares Linear Regression, Ridge and Lasso, each untuned, tuned with GridSearchCV and tuned with RandomizedSearchCV. They all reach a test R² of about 0.626 except untuned Lasso (alpha = 1), which scores 0.48.

Lasso + RandomizedSearchCV was chosen because:

- **It has the smallest train–test gap**, so it generalises the most consistently.
- **It is as accurate as the best model.** Its test RMSE is 8.610 against 8.605, a difference of 0.005 marks.
- **It is simpler to explain.** Lasso shrinks weak features towards zero, which leaves the factors that matter most.

Most of the improvement came from **feature engineering, not the model choice**. Adding `class_size`, `study_band` and the threshold flags raised test R² from 0.589 to 0.626.
