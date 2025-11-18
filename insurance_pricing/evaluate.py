from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }


def summarize_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df = df.sort_values("rmse")
    return df
