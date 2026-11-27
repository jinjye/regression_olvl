"""
Data preparation: load, clean and engineer features for the final_test model,
then split into train / test sets and build the preprocessing transformer.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Raw columns the cleaning and feature-engineering steps use. The target column (from config) is checked too.
REQUIRED_COLUMNS = [
    "tuition", "CCA", "age", "sleep_time", "wake_time", "n_male", "n_female",
    "hours_per_week", "attendance_rate", "learning_style", "direct_admission", "number_of_siblings",
]

# If cleaning removes more than this share of rows, log a warning (something may be wrong with the data)
MAX_SHARE_REMOVED = 0.20


class DataPreparation:
    def __init__(self, config: dict):
        self.config = config
        self.target = config["data"]["target"]
        self.numerical_features = config["features"]["numerical_features"]
        self.nominal_features = config["features"]["nominal_features"]
        self.fill_values = {}   # training medians used to fill missing values (saved with the model)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_data(self, file_path: str, require_target: bool = True) -> pd.DataFrame:
        """Read the CSV and check it has rows and all the columns the pipeline needs.
        New students to predict don't have a final_test yet, so require_target=False skips that check."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, index_col=[0])
        logger.info("Loaded %s rows x %s columns from %s", *df.shape, file_path)

        if len(df) == 0:
            raise ValueError(f"Data file {file_path} has no rows.")

        required = REQUIRED_COLUMNS + [self.target] if require_target else REQUIRED_COLUMNS
        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Data file {file_path} is missing these columns: {missing_columns}")

        logger.info("Schema check passed: all %s required columns are present", len(required))
        return df

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix inconsistent labels. Used for both training data and new data."""
        cfg = self.config["cleaning"]
        df = df.copy()
        df["tuition"] = df["tuition"].replace(cfg["tuition_mapping"])
        df["CCA"] = df["CCA"].str.capitalize().fillna(cfg["cca_fill_value"])
        df["age"] = df["age"].replace(cfg["age_mapping"])
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix labels, then drop duplicate, incomplete and invalid rows (training data only)."""
        cfg = self.config["cleaning"]
        df = self.fix_labels(df)

        # Drop duplicate / incomplete / invalid rows
        before = len(df)
        duplicates = df.duplicated().sum()   # exact copies of another row (same student recorded twice)
        df = df.drop_duplicates()
        logger.info("Dropped %s duplicate rows", duplicates)
        df = df.dropna(subset=cfg["dropna_columns"], how="any")
        df = df[df["age"] > cfg["min_age"]]
        logger.info("Cleaning removed %s rows, %s remain", before - len(df), len(df))

        if len(df) == 0:
            raise ValueError("No rows left after cleaning. Check the dropna_columns and min_age settings.")
        if (before - len(df)) / before > MAX_SHARE_REMOVED:
            logger.warning("Cleaning removed more than %.0f%% of the rows. Check the data.", MAX_SHARE_REMOVED * 100)
        return df

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config["features"]
        df = df.copy()

        # Hours of sleep, handling bedtimes before midnight (e.g. 22:00 -> 6:00 = 8 hrs)
        # errors="coerce" turns a badly written time (e.g. "abc" or "25:99") into NaT instead of crashing
        sleep = pd.to_timedelta(df["sleep_time"].astype(str) + ":00", errors="coerce")
        wake = pd.to_timedelta(df["wake_time"].astype(str) + ":00", errors="coerce")

        bad_times = sleep.isna() | wake.isna()
        if bad_times.any():
            logger.warning("Dropping %s rows with a missing or invalid sleep_time / wake_time", bad_times.sum())
            df, sleep, wake = df[~bad_times].copy(), sleep[~bad_times], wake[~bad_times]
        if len(df) == 0:
            raise ValueError("No rows left: every sleep_time / wake_time is missing or invalid.")

        df["sleep_hours"] = ((wake - sleep) % pd.Timedelta(days=1)).dt.total_seconds() / 3600

        # Class size (stronger predictor than n_male / n_female on their own)
        df["class_size"] = df["n_male"] + df["n_female"]

        # Study hours as bands, since the score peaks at 5-10 hrs
        df["study_band"] = pd.cut(
            df["hours_per_week"],
            bins=cfg["study_band"]["bins"],
            labels=cfg["study_band"]["labels"],
        ).astype(str)

        # Threshold flags found in the EDA
        # (missing attendance_rate -> low_attendance = 0, which matches the median fill of 95 done after the split)
        df["low_attendance"] = (df["attendance_rate"] <= cfg["low_attendance_threshold"]).astype(int)
        df["enough_sleep"] = (df["sleep_hours"] >= cfg["enough_sleep_threshold"]).astype(int)

        logger.info("Engineered features: sleep_hours, class_size, study_band, low_attendance, enough_sleep")
        return df

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    def split_data(self, df: pd.DataFrame):
        cfg = self.config["split"]

        # Every feature listed in config.yaml must exist (catches typos in the feature lists)
        missing_features = [col for col in self.numerical_features + self.nominal_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"These features in config.yaml don't exist in the data: {missing_features}")

        X = df[self.numerical_features + self.nominal_features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
        )
        logger.info("Split: %s train rows, %s test rows", len(X_train), len(X_test))

        # Fill missing values with the median of the TRAINING rows only (so the test set doesn't leak in)
        for column in self.config["cleaning"]["median_fill_columns"]:
            if column not in X_train.columns:
                raise ValueError(f"median_fill_columns has '{column}', which is not one of the model features.")
            median = X_train[column].median()
            self.fill_values[column] = median   # remembered so new data is filled the same way
            missing = X_train[column].isna().sum() + X_test[column].isna().sum()
            X_train = X_train.fillna({column: median})
            X_test = X_test.fillna({column: median})
            logger.info("Filled %s missing %s values with the training median (%s)", missing, column, median)

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # New data: prepare students to predict, the same way as the training data
    # ------------------------------------------------------------------
    def prepare_new_data(self, df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
        """Fix labels, build the features and fill missing values with the saved training medians.
        Rows are not dropped for a missing final_test, because new students don't have one yet."""
        df = self.fix_labels(df)
        df = self.engineer_features(df)

        missing_features = [col for col in self.numerical_features + self.nominal_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"These features in config.yaml don't exist in the new data: {missing_features}")

        X = df[self.numerical_features + self.nominal_features]
        X = X.fillna(fill_values)
        return X

    # ------------------------------------------------------------------
    # 5. Preprocessor (scaling + one-hot encoding, fitted later inside the model pipeline)
    # ------------------------------------------------------------------
    def create_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numerical_features),
            ("nom", OneHotEncoder(handle_unknown="ignore", drop="first"), self.nominal_features),
        ])

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------
    def run(self, file_path: str):
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.engineer_features(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        preprocessor = self.create_preprocessor()
        return X_train, X_test, y_train, y_test, preprocessor
