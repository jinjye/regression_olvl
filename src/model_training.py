"""
Model training: Random forest tuned with RandomizedSearchCV,
compared with a mean baseline, evaluated on the held-out test set, and saved to disk.
"""

import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTraining:
    def __init__(self, config: dict, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.search = None
        self.best_model = None

    # ------------------------------------------------------------------
    # 1. Tune the random forest with RandomizedSearchCV (cross-validation on training set only)
    # ------------------------------------------------------------------
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
        """Try random combinations of forest settings with k-fold CV and keep the best model."""
        cfg = self.config["model"]
        search_cfg = cfg["random_search"]

        # Cross-validation needs at least as many training rows as folds
        if len(X_train) < search_cfg["cv"]:
            raise ValueError(f"Only {len(X_train)} training rows, but cv = {search_cfg['cv']} folds. "
                             "Add more data or lower model.random_search.cv in config.yaml.")

        pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", RandomForestRegressor(random_state=cfg["random_state"])),
        ])

        # The pipeline step is called "regressor", so each setting needs the "regressor__" prefix
        param_distributions = {}
        for name, values in cfg["param_distributions"].items():
            param_distributions["regressor__" + name] = values

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=search_cfg["n_iter"],
            cv=search_cfg["cv"],
            scoring=search_cfg["scoring"],
            random_state=search_cfg["random_state"],
            n_jobs=search_cfg["n_jobs"],
        )

        logger.info("Running RandomizedSearchCV (%s candidates x %s folds)...",
                    search_cfg["n_iter"], search_cfg["cv"])
        search.fit(X_train, y_train)

        self.search = search
        self.best_model = search.best_estimator_
        logger.info("Best parameters: %s", search.best_params_)
        logger.info("Best CV %s: %.4f", search_cfg["scoring"], search.best_score_)
        return search

    # ------------------------------------------------------------------
    # 2. Fold-to-fold variation of the best model
    # ------------------------------------------------------------------
    def cv_summary(self) -> pd.DataFrame:
        """Score of the best model on each CV fold, plus the mean and standard deviation."""
        if self.search is None:
            raise RuntimeError("Call tune_model() before cv_summary().")

        best = self.search.best_index_
        n_folds = self.config["model"]["random_search"]["cv"]
        fold_scores = [self.search.cv_results_[f"split{i}_test_score"][best] for i in range(n_folds)]

        summary = pd.DataFrame({"fold": [f"fold {i + 1}" for i in range(n_folds)], "score": fold_scores})
        summary.loc[len(summary)] = ["mean", np.mean(fold_scores)]
        summary.loc[len(summary)] = ["std", np.std(fold_scores)]
        return summary

    # ------------------------------------------------------------------
    # 3. Evaluate on train and test sets, next to a mean baseline
    # ------------------------------------------------------------------
    @staticmethod
    def _metrics(y_true, y_pred) -> dict:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    def evaluate_model(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """Train and test metrics for the best model, plus a baseline that always predicts the training mean."""
        if self.best_model is None:
            raise RuntimeError("Call tune_model() before evaluate_model().")

        baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

        results = pd.DataFrame({
            "Train": self._metrics(y_train, self.best_model.predict(X_train)),
            "Test": self._metrics(y_test, self.best_model.predict(X_test)),
            "Baseline (mean) Test": self._metrics(y_test, baseline.predict(X_test)),
        })
        results["Gap (test - train)"] = results["Test"] - results["Train"]
        return results

    # ------------------------------------------------------------------
    # 4. Feature importances (which features the forest relies on most)
    # ------------------------------------------------------------------
    def get_feature_importances(self) -> pd.Series:
        if self.best_model is None:
            raise RuntimeError("Call tune_model() before get_feature_importances().")

        names = self.best_model.named_steps["preprocessor"].get_feature_names_out()
        names = [n.replace("num__", "").replace("nom__", "") for n in names]
        importances = pd.Series(self.best_model.named_steps["regressor"].feature_importances_, index=names)
        return importances.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # 5. Save everything needed to reproduce, audit and reuse the run
    # ------------------------------------------------------------------
    def save_outputs(self, results: pd.DataFrame, X_train, X_test, y_test, fill_values: dict) -> Path:
        """Save the model, metrics, CV folds, test predictions, feature importances, config and a run record."""
        if self.best_model is None:
            raise RuntimeError("Call tune_model() before save_outputs().")

        out_dir = Path(self.config["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # The fitted pipeline (preprocessor + forest), so it can be loaded later to predict new students
        joblib.dump(self.best_model, out_dir / "model.joblib")

        # Tables
        results.to_csv(out_dir / "metrics.csv")
        self.cv_summary().to_csv(out_dir / "cv_folds.csv", index=False)
        self.get_feature_importances().to_frame("importance").to_csv(out_dir / "feature_importances.csv")

        predictions = pd.DataFrame({"actual": y_test, "predicted": self.best_model.predict(X_test)})
        predictions["error"] = predictions["predicted"] - predictions["actual"]
        predictions.to_csv(out_dir / "test_predictions.csv")

        # A copy of the exact config used for this run
        with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        # A machine-readable record of the selected model and why it was selected
        cv = self.cv_summary().set_index("fold")["score"]
        record = {
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "model": "RandomForestRegressor + RandomizedSearchCV",
            "best_params": {k.replace("regressor__", ""): v for k, v in self.search.best_params_.items()},
            "cv_scoring": self.config["model"]["random_search"]["scoring"],
            "cv_mean": float(cv["mean"]),
            "cv_std": float(cv["std"]),
            "test_metrics": {k: float(v) for k, v in results["Test"].items()},
            "baseline_test_metrics": {k: float(v) for k, v in results["Baseline (mean) Test"].items()},
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fill_values": {k: float(v) for k, v in fill_values.items()},   # needed to prepare new data the same way
            "numerical_features": self.config["features"]["numerical_features"],
            "nominal_features": self.config["features"]["nominal_features"],
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
        }
        with open(out_dir / "model_record.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        logger.info("Saved model, metrics, CV folds, predictions, feature importances and run record to %s/",
                    out_dir)
        return out_dir
