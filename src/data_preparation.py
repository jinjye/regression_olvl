"""
Data preparation: load, clean and engineer features for the final_test model,
then split into train / test sets and build the preprocessing transformer.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(self, config: dict):
        self.config = config
        self.target = config["data"]["target"]
        self.numerical_features = config["features"]["numerical_features"]
        self.nominal_features = config["features"]["nominal_features"]

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, index_col=[0])
        logger.info("Loaded %s rows x %s columns from %s", *df.shape, file_path)
        return df

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config["cleaning"]
        df = df.copy()

        # Fix inconsistent labels
        df["tuition"] = df["tuition"].replace(cfg["tuition_mapping"])
        df["CCA"] = df["CCA"].str.capitalize().fillna(cfg["cca_fill_value"])
        df["age"] = df["age"].replace(cfg["age_mapping"])

        # Drop incomplete / invalid rows
        before = len(df)
        df = df.dropna(subset=cfg["dropna_columns"], how="any")
        df = df[df["age"] > cfg["min_age"]]
        logger.info("Cleaning removed %s rows, %s remain", before - len(df), len(df))
        return df

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config["features"]
        df = df.copy()

        # Hours of sleep, handling bedtimes before midnight (e.g. 22:00 -> 6:00 = 8 hrs)
        sleep = pd.to_timedelta(df["sleep_time"] + ":00")
        wake = pd.to_timedelta(df["wake_time"] + ":00")
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
        df["low_attendance"] = (df["attendance_rate"] <= cfg["low_attendance_threshold"]).astype(int)
        df["enough_sleep"] = (df["sleep_hours"] >= cfg["enough_sleep_threshold"]).astype(int)

        logger.info("Engineered features: sleep_hours, class_size, study_band, low_attendance, enough_sleep")
        return df

    # ------------------------------------------------------------------
    # 4. Split
    # ------------------------------------------------------------------
    def split_data(self, df: pd.DataFrame):
        cfg = self.config["split"]
        X = df[self.numerical_features + self.nominal_features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
        )
        logger.info("Split: %s train rows, %s test rows", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

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
