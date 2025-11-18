from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# Optional advanced libs
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception:  # pragma: no cover
    optuna = None
    TPESampler = None


def get_model_candidates(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Provide a lightweight registry of models for the training script.
    Reuses the baseline specs so train.py can stay simple.
    """
    baseline = SafeBaselineModels(random_state=random_state)
    return baseline._build_models()


# --------------------------------------------------------------------
# Metric utility
# --------------------------------------------------------------------
def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "r2": r2, "mae": mae}


# --------------------------------------------------------------------
# Baseline models (mirrors notebook SafeBaselineModels)
# --------------------------------------------------------------------
class SafeBaselineModels:
    """
    Train baseline models with proper regularization to reduce overfitting.
    Models reflect the notebook:
      - Linear Regression
      - Ridge, Lasso, ElasticNet (more regularized)
      - DecisionTreeRegressor (shallow, min samples)
      - RandomForestRegressor (reduced depth/estimators)
      - GradientBoostingRegressor (conservative settings)
      - KNeighborsRegressor (more neighbors, distance weights)
      - SVR (rbf with moderate C)
    """

    def __init__(self, cv_splits: int = 5, random_state: int = 42):
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.models: Dict[str, BaseEstimator] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def _build_models(self) -> Dict[str, BaseEstimator]:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0, random_state=self.random_state),
            "Lasso Regression": Lasso(alpha=1.0, random_state=self.random_state),
            "ElasticNet": ElasticNet(
                alpha=1.0, l1_ratio=0.5, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.random_state,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=self.random_state,
            ),
            "K-Neighbors": KNeighborsRegressor(
                n_neighbors=20,
                weights="distance",
            ),
            "SVR": SVR(
                kernel="rbf",
                C=100,
                gamma="scale",
            ),
        }
        return models

    def train_baseline_models(
        self, X_train, y_train, X_val, y_val
    ) -> pd.DataFrame:
        """
        Mirrors the notebook logic:
        - Train each baseline
        - Compute train / val metrics
        - Do cross-val RMSE for robustness
        - Track overfitting gap
        """
        print("\n🤖 TRAINING REGULARIZED BASELINE MODELS")
        print("=" * 80)

        models = self._build_models()
        self.models = {}
        self.results = {}

        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for name, model in models.items():
            print(f"\n▶ Training {name}...")
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Metrics
            train_metrics = _regression_metrics(y_train, y_train_pred)
            val_metrics = _regression_metrics(y_val, y_val_pred)

            # Cross-val on train
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse = -cv_scores.mean()
            cv_std = cv_scores.std()

            overfitting_rmse = train_metrics["rmse"] - val_metrics["rmse"]
            overfitting_r2 = train_metrics["r2"] - val_metrics["r2"]

            self.results[name] = {
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "cv_rmse": cv_rmse,
                "cv_std": cv_std,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "overfitting_rmse": overfitting_rmse,
                "overfitting_r2": overfitting_r2,
                "train_time": train_time,
            }

            self.models[name] = model

            print(
                f"  Train RMSE: {train_metrics['rmse']:.2f} | "
                f"Val RMSE: {val_metrics['rmse']:.2f} | "
                f"CV RMSE: {cv_rmse:.2f} ± {cv_std:.2f}"
            )
            print(
                f"  Train R²: {train_metrics['r2']:.4f} | "
                f"Val R²: {val_metrics['r2']:.4f} | "
                f"Overfit ΔR²: {overfitting_r2:.4f}"
            )

        df = pd.DataFrame(self.results).T.sort_values("val_rmse")
        print("\n📊 Baseline Model Summary (sorted by Val RMSE)")
        print(df[["train_rmse", "val_rmse", "train_r2", "val_r2", "cv_rmse", "cv_std"]])
        return df

    def get_models(self) -> Dict[str, BaseEstimator]:
        return self.models

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).T


# --------------------------------------------------------------------
# Advanced models (mirrors notebook SafeAdvancedModels)
# --------------------------------------------------------------------
class SafeAdvancedModels:
    """
    Train advanced models (XGBoost / LightGBM / CatBoost) with strong
    regularization and optional Optuna tuning, consistent with the notebook.
    """

    def __init__(self, use_optuna: bool = False, n_trials: int = 20):
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.models: Dict[str, BaseEstimator] = {}
        self.best_params: Dict[str, Dict] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    # ----------------------- XGBoost -----------------------
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("\n🎯 Training XGBoost with Regularization...")
        print("-" * 50)

        if xgb is None:
            print("⚠️ xgboost not installed. Skipping XGBoost.")
            return None

        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        X_train_xgb.columns = [str(c) for c in X_train_xgb.columns]
        X_val_xgb.columns = [str(c) for c in X_val_xgb.columns]

        if self.use_optuna:
            if optuna is None or TPESampler is None:
                raise ImportError(
                    "optuna is not available. Install it or set use_optuna=False."
                )

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.1, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 0.8
                    ),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight", 5, 20
                    ),
                    "random_state": 42,
                }
                model = xgb.XGBRegressor(**params)
                cv_scores = cross_val_score(
                    model,
                    X_train_xgb,
                    y_train,
                    cv=3,
                    scoring="neg_root_mean_squared_error",
                )
                return -cv_scores.mean()

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="minimize", sampler=TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=self.n_trials)

            print(f"Best XGBoost params: {study.best_params}")
            params = {
                **study.best_params,
                "random_state": 42,
                "n_estimators": max(50, study.best_params.get("n_estimators", 100)),
            }

        else:
            params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "min_child_weight": 10,
                "random_state": 42,
            }

        model = xgb.XGBRegressor(**params)

        start = time.time()
        model.fit(
            X_train_xgb,
            y_train,
            eval_set=[(X_val_xgb, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=30,
            verbose=False,
        )
        train_time = time.time() - start

        y_train_pred = model.predict(X_train_xgb)
        y_val_pred = model.predict(X_val_xgb)

        train_metrics = _regression_metrics(y_train, y_train_pred)
        val_metrics = _regression_metrics(y_val, y_val_pred)

        self.models["XGBoost"] = model
        self.best_params["XGBoost"] = params
        self.results["XGBoost"] = {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_time": train_time,
        }

        print(
            f"  Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        return model

    # ----------------------- LightGBM -----------------------
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("\n🌿 Training LightGBM with Regularization...")
        print("-" * 50)

        if lgb is None:
            print("⚠️ lightgbm not installed. Skipping LightGBM.")
            return None

        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "random_state": 42,
        }

        model = lgb.LGBMRegressor(**params)
        start = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        train_time = time.time() - start

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics = _regression_metrics(y_train, y_train_pred)
        val_metrics = _regression_metrics(y_val, y_val_pred)

        self.models["LightGBM"] = model
        self.best_params["LightGBM"] = params
        self.results["LightGBM"] = {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_time": train_time,
        }

        print(
            f"  Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        return model

    # ----------------------- CatBoost -----------------------
    def train_catboost(self, X_train, y_train, X_val, y_val):
        print("\n🐱 Training CatBoost with Regularization...")
        print("-" * 50)

        if CatBoostRegressor is None:
            print("⚠️ catboost not installed. Skipping CatBoost.")
            return None

        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            iterations=500,
            random_state=42,
            loss_function="RMSE",
            verbose=False,
        )

        start = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        train_time = time.time() - start

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics = _regression_metrics(y_train, y_train_pred)
        val_metrics = _regression_metrics(y_val, y_val_pred)

        self.models["CatBoost"] = model
        self.best_params["CatBoost"] = {
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "iterations": 500,
        }
        self.results["CatBoost"] = {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_time": train_time,
        }

        print(
            f"  Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        return model

    # ----------------------- Orchestrator -----------------------
    def train_all(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        """
        Convenience wrapper to train all advanced models.
        """
        if xgb is not None:
            self.train_xgboost(X_train, y_train, X_val, y_val)
        if lgb is not None:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
        if CatBoostRegressor is not None:
            self.train_catboost(X_train, y_train, X_val, y_val)

        df = pd.DataFrame(self.results).T.sort_values("val_rmse")
        print("\n📊 Advanced Model Summary (sorted by Val RMSE)")
        print(df[["train_rmse", "val_rmse", "train_r2", "val_r2"]])
        return df


# --------------------------------------------------------------------
# Ensemble methods (mirrors ProperEnsembleMethods)
# --------------------------------------------------------------------
class ProperEnsembleMethods:
    """
    Corrected ensemble implementation that avoids leakage and
    correctly implements voting and stacking over base model specs.
    """

    def __init__(self, base_model_specs: Optional[Dict[str, BaseEstimator]] = None):
        """
        base_model_specs: dict of {name: untrained_model_instance}
        """
        self.base_model_specs = base_model_specs if base_model_specs else {}
        self.ensemble_models: Dict[str, BaseEstimator] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    # ----------------------- Voting -----------------------
    def create_voting_ensemble(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model_names: Optional[List[str]] = None,
    ) -> VotingRegressor:
        print("\n🗳️ Creating Voting Ensemble...")
        print("-" * 50)

        if model_names is None:
            model_names = list(self.base_model_specs.keys())

        estimators: List[Tuple[str, BaseEstimator]] = []
        for name in model_names:
            if name in self.base_model_specs:
                print(f"  Training fresh {name} for voting...")
                fresh_model = clone(self.base_model_specs[name])
                fresh_model.fit(X_train, y_train)
                estimators.append((name, fresh_model))

        voting = VotingRegressor(estimators=estimators, n_jobs=-1)
        voting.fit(X_train, y_train)

        y_train_pred = voting.predict(X_train)
        y_val_pred = voting.predict(X_val)

        train_metrics = _regression_metrics(y_train, y_train_pred)
        val_metrics = _regression_metrics(y_val, y_val_pred)

        self.ensemble_models["Voting"] = voting
        self.results["Voting"] = {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
        }

        print(
            f"  Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        return voting

    # ----------------------- Stacking -----------------------
    def create_stacking_ensemble(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        model_names: Optional[List[str]] = None,
        final_estimator: Optional[BaseEstimator] = None,
    ) -> StackingRegressor:
        print("\n🏗️ Creating Stacking Ensemble...")
        print("-" * 50)

        if model_names is None:
            model_names = list(self.base_model_specs.keys())

        if final_estimator is None:
            final_estimator = Ridge(alpha=1.0, random_state=42)

        estimators: List[Tuple[str, BaseEstimator]] = []
        for name in model_names:
            if name in self.base_model_specs:
                print(f"  Preparing base model {name} for stacking...")
                estimators.append((name, clone(self.base_model_specs[name])))

        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
        )
        stack.fit(X_train, y_train)

        y_train_pred = stack.predict(X_train)
        y_val_pred = stack.predict(X_val)

        train_metrics = _regression_metrics(y_train, y_train_pred)
        val_metrics = _regression_metrics(y_val, y_val_pred)

        self.ensemble_models["Stacking"] = stack
        self.results["Stacking"] = {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
        }

        print(
            f"  Train RMSE: {train_metrics['rmse']:.2f} | "
            f"Val RMSE: {val_metrics['rmse']:.2f} | "
            f"Val R²: {val_metrics['r2']:.4f}"
        )
        return stack

    def get_ensemble_models(self) -> Dict[str, BaseEstimator]:
        return self.ensemble_models

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).T


# --------------------------------------------------------------------
# Comprehensive comparison (mirrors ComprehensiveModelComparison)
# --------------------------------------------------------------------
class ComprehensiveModelComparison:
    """
    Evaluate and compare multiple models (baseline, advanced, ensemble)
    on a common test set, consistent with the notebook’s intent.
    """

    def __init__(
        self,
        models_dict: Dict[str, BaseEstimator],
        X_test,
        y_test,
        model_categories: Optional[Dict[str, str]] = None,
    ):
        """
        models_dict: {name: fitted_model}
        model_categories: optional {name: category}
          categories examples: 'Linear', 'Tree-Based', 'Neural Network',
                               'Ensemble', 'Other'
        """
        self.models = models_dict
        self.X_test = X_test
        self.y_test = y_test
        self.model_categories = (
            model_categories if model_categories else self._auto_categorize_models()
        )
        self.results: Dict[str, Dict[str, float]] = {}

    def _auto_categorize_models(self) -> Dict[str, str]:
        categories = {
            "Linear": [],
            "Tree-Based": [],
            "Neural Network": [],
            "Ensemble": [],
            "Other": [],
        }
        mapping: Dict[str, str] = {}

        for name, model in self.models.items():
            mstr = str(type(model)).lower()
            if any(k in mstr for k in ["linearregression", "ridge", "lasso", "elasticnet"]):
                cat = "Linear"
            elif any(k in mstr for k in ["forest", "gradientboosting", "tree"]):
                cat = "Tree-Based"
            elif any(k in mstr for k in ["xgb", "lgbm", "catboost"]):
                cat = "Tree-Based"
            elif any(k in mstr for k in ["mlpregressor", "keras", "torch"]):
                cat = "Neural Network"
            elif any(k in mstr for k in ["votingregressor", "stackingregressor"]):
                cat = "Ensemble"
            else:
                cat = "Other"

            categories[cat].append(name)
            mapping[name] = cat

        return mapping

    def run_evaluation(self) -> pd.DataFrame:
        print("\n📊 Running Comprehensive Test-Set Evaluation")
        print("=" * 80)

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            start = time.time()
            y_pred = model.predict(self.X_test)
            pred_time = time.time() - start

            metrics = _regression_metrics(self.y_test, y_pred)
            metrics["prediction_time"] = pred_time
            metrics["category"] = self.model_categories.get(name, "Other")

            self.results[name] = metrics

        df = pd.DataFrame(self.results).T
        df = df.sort_values("rmse")

        print("\n✅ Evaluation Complete. Top models by RMSE:")
        print(df[["rmse", "r2", "mae", "prediction_time", "category"]].head())

        return df
    
