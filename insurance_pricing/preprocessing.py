import numpy as np
from typing import List, Tuple

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureEngineering


def _build_column_transformer() -> ColumnTransformer:
    """Build the core column transformer for numeric/categorical features."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_exclude=np.number)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_selector),
            ("cat", categorical_transformer, categorical_selector),
        ],
    )


def build_preprocessing_pipeline(
    include_feature_engineering: bool = True, verbose: bool = True
) -> Pipeline:
    """Return a preprocessing pipeline with optional FeatureEngineering."""
    steps: List[Tuple[str, BaseEstimator]] = []

    if include_feature_engineering:
        steps.append(("feature_engineering", FeatureEngineering(verbose=verbose)))

    steps.append(("preprocessing", _build_column_transformer()))

    return Pipeline(steps=steps)


def build_full_pipeline(model: BaseEstimator, verbose: bool = True) -> Pipeline:
    preprocessing_pipeline = build_preprocessing_pipeline(
        include_feature_engineering=True, verbose=verbose
    )

    steps = list(preprocessing_pipeline.steps)
    steps.append(("model", model))

    return Pipeline(steps=steps)
