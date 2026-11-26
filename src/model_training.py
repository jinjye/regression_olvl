"""
Model training: Lasso regression tuned with RandomizedSearchCV,
evaluated on the held-out test set.
"""

import logging

import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTraining:
    def __init__(self, config: dict, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.best_model = None

    # ------------------------------------------------------------------
    # 1. Tune Lasso with RandomizedSearchCV (cross-validation on training set only)
    # ------------------------------------------------------------------
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
        cfg = self.config["model"]
        search_cfg = cfg["random_search"]
        dist_cfg = cfg["param_distributions"]

        pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", Lasso(max_iter=cfg["max_iter"])),
        ])

        param_distributions = {
            "regressor__alpha": loguniform(dist_cfg["alpha"]["low"], dist_cfg["alpha"]["high"]),
            "regressor__fit_intercept": dist_cfg["fit_intercept"],
        }

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

        self.best_model = search.best_estimator_
        logger.info("Best parameters: %s", search.best_params_)
        logger.info("Best CV %s: %.4f", search_cfg["scoring"], search.best_score_)
        return search

    # ------------------------------------------------------------------
    # 2. Evaluate on train and test sets
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
        if self.best_model is None:
            raise RuntimeError("Call tune_model() before evaluate_model().")

        results = pd.DataFrame({
            "Train": self._metrics(y_train, self.best_model.predict(X_train)),
            "Test": self._metrics(y_test, self.best_model.predict(X_test)),
        })
        results["Gap (test - train)"] = results["Test"] - results["Train"]
        return results

    # ------------------------------------------------------------------
    # 3. Coefficients (which features the model kept)
    # ------------------------------------------------------------------
    def get_coefficients(self) -> pd.Series:
        names = self.best_model.named_steps["preprocessor"].get_feature_names_out()
        names = [n.replace("num__", "").replace("nom__", "") for n in names]
        coefs = pd.Series(self.best_model.named_steps["regressor"].coef_, index=names)
        return coefs.sort_values(key=abs, ascending=False)
