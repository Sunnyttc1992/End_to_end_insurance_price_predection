import joblib
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict

from .config import (
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    PROCESSED_DIR,
)
from .data_loader import load_insurance_data
from .models import get_model_candidates
from .preprocessing import build_full_pipeline
from .evaluate import regression_metrics, summarize_results


def train_and_select_model() -> None:
    df = load_insurance_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = get_model_candidates()
    results: Dict[str, Dict[str, float]] = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, base_model in models.items():
        print(f"\n=== Training model: {name} ===")
        pipeline = build_full_pipeline(base_model)

        # Cross-val predictions on TRAIN only (no leakage)
        y_train_pred = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
        )

        metrics = regression_metrics(y_train, y_train_pred)
        results[name] = metrics
        print(metrics)

    summary_df = summarize_results(results)
    print("\n=== CV Results (sorted by RMSE) ===")
    print(summary_df)

    # Choose best by RMSE
    best_model_name = summary_df.index[0]
    print(f"\nBest model by RMSE: {best_model_name}")

    best_model = get_model_candidates()[best_model_name]
    best_pipeline = build_full_pipeline(best_model)

    # Fit on full TRAIN data
    best_pipeline.fit(X_train, y_train)

    # Final hold-out evaluation
    y_test_pred = best_pipeline.predict(X_test)
    final_metrics = regression_metrics(y_test, y_test_pred)
    print("\n=== Final Test Metrics ===")
    print(final_metrics)

    # Persist model
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = PROCESSED_DIR / f"best_model_{best_model_name}.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best model to: {model_path.resolve()}")


if __name__ == "__main__":
    train_and_select_model()
