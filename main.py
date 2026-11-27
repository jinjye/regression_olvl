"""
Run the full pipeline: data preparation -> Random forest + RandomizedSearchCV -> evaluation -> save outputs.
Or load the saved model and predict final_test for new students.

Usage (from the project root):
    python main.py                                   # train, evaluate and save to outputs/
    python main.py --config src/config.yaml          # same, with a different config file
    python main.py --predict data/new_students.csv   # predict with the saved model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_preparation import DataPreparation  # noqa: E402
from model_training import ModelTraining  # noqa: E402


# Every key the pipeline reads from config.yaml, grouped by section.
# If any of these is missing or misspelled, the script stops at the start with a clear message.
REQUIRED_CONFIG_KEYS = {
    "data": ["file_path", "target"],
    "cleaning": ["tuition_mapping", "cca_fill_value", "age_mapping", "dropna_columns", "min_age",
                 "median_fill_columns"],
    "features": ["study_band", "low_attendance_threshold", "enough_sleep_threshold",
                 "numerical_features", "nominal_features"],
    "split": ["test_size", "random_state"],
    "model": ["random_state", "random_search", "param_distributions"],
    "output": ["dir"],
}


def load_config(config_path: Path) -> dict:
    """Read config.yaml and return it as a dictionary. Stops with a clear message if it can't be read."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as error:
        raise ValueError(f"Config file {config_path} is not valid YAML:\n{error}")

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is empty or not in 'key: value' format.")
    return config


def validate_config(config: dict) -> None:
    """Check that every required key is in the config and that key values make sense."""
    # 1. Every required section and key exists
    missing = []
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            missing.append(f"section '{section}'")
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"'{section}.{key}'")
    if missing:
        raise ValueError("config.yaml is missing: " + ", ".join(missing))

    # 2. Values are sensible
    test_size = config["split"]["test_size"]
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError(f"split.test_size must be a number between 0 and 1, got {test_size!r}")

    for key in ["numerical_features", "nominal_features"]:
        value = config["features"][key]
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError(f"features.{key} must be a non-empty list of column names, got {value!r}")

    for key in ["low_attendance_threshold", "enough_sleep_threshold"]:
        value = config["features"][key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"features.{key} must be a number, got {value!r}")

    for key in ["n_iter", "cv", "scoring", "random_state", "n_jobs"]:
        if key not in config["model"]["random_search"]:
            raise ValueError(f"config.yaml is missing 'model.random_search.{key}'")
    cv = config["model"]["random_search"]["cv"]
    if not isinstance(cv, int) or cv < 2:
        raise ValueError(f"model.random_search.cv must be a whole number of at least 2, got {cv!r}")

    logging.info("Config check passed")


def main(config_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.chdir(PROJECT_ROOT)  # so relative paths in config.yaml resolve from the project root

    config = load_config(config_path)
    validate_config(config)

    # 1. Data preparation
    data_prep = DataPreparation(config)
    X_train, X_test, y_train, y_test, preprocessor = data_prep.run(config["data"]["file_path"])

    # 2. Model training
    trainer = ModelTraining(config, preprocessor)
    trainer.tune_model(X_train, y_train)

    # 3. Evaluation
    results = trainer.evaluate_model(X_train, y_train, X_test, y_test)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n================ Random forest + RandomizedSearchCV ================")
    print(results)
    print(f"\nCross-validation {config['model']['random_search']['scoring']} on each fold (training set):")
    print(trainer.cv_summary().to_string(index=False))
    print("\nTop 10 features by importance:")
    print(trainer.get_feature_importances().head(10).to_string())

    # 4. Save the model and results
    out_dir = trainer.save_outputs(results, X_train, X_test, y_test, data_prep.fill_values)
    print(f"\nSaved to {out_dir}/: model.joblib, metrics.csv, cv_folds.csv, test_predictions.csv, "
          "feature_importances.csv, config_used.yaml, model_record.json")


def predict(config_path: Path, new_data_path: Path) -> None:
    """Load the saved model and predict final_test for new students. Saves outputs/predictions.csv."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    new_data_path = Path(new_data_path).resolve()   # resolve before changing folder
    os.chdir(PROJECT_ROOT)

    config = load_config(config_path)
    validate_config(config)
    out_dir = Path(config["output"]["dir"])

    model_path = out_dir / "model.joblib"
    record_path = out_dir / "model_record.json"
    if not model_path.exists() or not record_path.exists():
        raise FileNotFoundError(f"No saved model in {out_dir}/. Run 'python main.py' first to train one.")

    model = joblib.load(model_path)
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # Prepare the new students exactly like the training data, using the saved training medians
    data_prep = DataPreparation(config)
    df = data_prep.load_data(new_data_path, require_target=False)
    X_new = data_prep.prepare_new_data(df, record["fill_values"])

    predictions = pd.DataFrame({"predicted_final_test": model.predict(X_new).round(1)}, index=X_new.index)
    if "student_id" in df.columns:
        predictions.insert(0, "student_id", df.loc[X_new.index, "student_id"])

    predictions.to_csv(out_dir / "predictions.csv")
    print(predictions.head(10).to_string())
    print(f"\nPredicted {len(predictions)} students. Saved to {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final_test prediction model, or predict new students.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src" / "config.yaml",
                        help="Path to the YAML config file (default: src/config.yaml)")
    parser.add_argument("--predict", type=Path, default=None,
                        help="CSV of new students to predict with the saved model (skips training)")
    args = parser.parse_args()

    if args.predict:
        predict(args.config, args.predict)
    else:
        main(args.config)
