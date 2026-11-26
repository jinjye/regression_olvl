"""
Run the full pipeline: data preparation -> Lasso + RandomizedSearchCV -> evaluation.

Usage (from the project root):
    python main.py
    python main.py --config src/config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_preparation import DataPreparation  # noqa: E402
from model_training import ModelTraining  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.chdir(PROJECT_ROOT)  # so relative paths in config.yaml resolve from the project root

    config = load_config(config_path)

    # 1. Data preparation
    data_prep = DataPreparation(config)
    X_train, X_test, y_train, y_test, preprocessor = data_prep.run(config["data"]["file_path"])

    # 2. Model training
    trainer = ModelTraining(config, preprocessor)
    trainer.tune_model(X_train, y_train)

    # 3. Evaluation
    results = trainer.evaluate_model(X_train, y_train, X_test, y_test)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n================ Lasso + RandomizedSearchCV ================")
    print(results)
    print("\nTop 10 features by coefficient size:")
    print(trainer.get_coefficients().head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final_test prediction model.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "src" / "config.yaml",
                        help="Path to the YAML config file (default: src/config.yaml)")
    args = parser.parse_args()
    main(args.config)
