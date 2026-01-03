import os
import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PredictionInput(BaseModel):
    age: int = Field(..., ge=0)
    sex: str
    bmi: float = Field(..., ge=0)
    children: int = Field(..., ge=0)
    smoker: str
    region: str


class PredictionRequest(BaseModel):
    records: List[PredictionInput]


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config file not found at %s; using defaults.", config_path)
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def _resolve_model_path(config: dict, project_root: Path) -> Path:
    model_name = config.get("model", {}).get("name", "insurance_price_model")
    default_path = project_root / "models" / "trained" / f"{model_name}.pkl"
    fallback_path = project_root / "artifacts" / "trained" / f"{model_name}.pkl"
    if default_path.exists():
        return default_path
    if fallback_path.exists():
        return fallback_path
    return default_path


def _load_feature_columns(training_data_path: Path, target_column: str) -> List[str]:
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_data_path}")
    df = pd.read_csv(training_data_path)
    X = df.drop(columns=[target_column], errors="ignore")
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=object_cols, drop_first=True)
    return X.columns.tolist()


def _prepare_features(input_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    object_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(input_df, columns=object_cols, drop_first=True)
    return X.reindex(columns=feature_columns, fill_value=0)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "configs" / "model_config.yaml"))
TRAINING_DATA_PATH = Path(
    os.getenv("TRAINING_DATA_PATH", PROJECT_ROOT / "data" / "processed" / "clean_insurance.csv")
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", ""))

config = _load_config(CONFIG_PATH)
target_variable = config.get("model", {}).get("target_variable", "charges")

resolved_model_path = MODEL_PATH if MODEL_PATH else _resolve_model_path(config, PROJECT_ROOT)

try:
    model = joblib.load(resolved_model_path)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Model file not found at {resolved_model_path}. "
        "Set MODEL_PATH or train the model first."
    ) from exc

try:
    feature_columns = _load_feature_columns(TRAINING_DATA_PATH, target_variable)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Training data not found at {TRAINING_DATA_PATH}. "
        "Set TRAINING_DATA_PATH to the processed training data."
    ) from exc

app = FastAPI(title="Insurance Price Prediction API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")
    input_df = pd.DataFrame([r.model_dump() for r in request.records])
    features = _prepare_features(input_df, feature_columns)
    preds = model.predict(features)
    return {"predictions": [float(p) for p in preds]}
