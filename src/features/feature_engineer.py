import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("feature-engineering")


# ---------------------------------------------------------------------
# Feature engineering logic
# ---------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the insurance dataset.

    Required columns:
      - age, bmi, children, smoker, region
    """
    logger.info("Starting feature engineering")

    required = {"age", "bmi", "children", "smoker", "region"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    logger.info("Input shape: %s", df.shape)

    # 1) Risk-oriented derived features
    logger.info("Creating age_group and bmi_category")
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    )

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True,
    )

    df["is_smoker"] = (df["smoker"].astype(str).str.lower() == "yes").astype(int)
    df["high_bmi"] = (df["bmi"] >= 30).astype(int)

    # 2) Interaction features
    logger.info("Creating interaction features")
    df["age_smoker_interaction"] = df["age"] * df["is_smoker"]
    df["bmi_smoker_interaction"] = df["bmi"] * df["is_smoker"]

    df["family_size"] = df["children"] + 1
    df["large_family"] = (df["family_size"] >= 4).astype(int)

    # 3) Regional risk normalization
    logger.info("Creating region-based features")
    region_freq = df["region"].value_counts(normalize=True)
    df["region_freq"] = df["region"].map(region_freq)

    df["region_smoker"] = df["region"].astype(str) + "_" + df["smoker"].astype(str)

    # 4) Behavioral / economic proxies
    logger.info("Creating behavioral indicators")
    df["is_adult"] = (df["age"] >= 18).astype(int)
    df["prime_risk_age"] = ((df["age"] >= 40) & (df["age"] <= 60)).astype(int)

    logger.info("Feature engineering completed")
    logger.info("Output shape: %s", df.shape)

    return df


def create_preprocessor(df: pd.DataFrame, target_col: str = "charges") -> ColumnTransformer:
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessor pipeline")

    # Define feature groups by dtype
    categorical_features = (
        df.select_dtypes(include=["object", "string", "category"])
          .columns
          .drop(target_col, errors="ignore")
          .tolist()
    )

    numerical_features = (
        df.select_dtypes(include=["number", "bool"])
          .columns
          .drop(target_col, errors="ignore")
          .tolist()
    )

    logger.info("Numeric features (%d): %s", len(numerical_features), numerical_features)
    logger.info("Categorical features (%d): %s", len(categorical_features), categorical_features)

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Best-effort feature name extraction after fitting."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            try:
                ohe = trans.named_steps["onehot"]
                names.extend(ohe.get_feature_names_out(cols).tolist())
            except Exception:
                # fallback
                names.extend([f"{c}_ohe" for c in cols])
    return names


def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full feature engineering pipeline."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    preprocessor_path = Path(preprocessor_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path)

    # Create features
    df_featured = create_features(df)
    logger.info("Created featured dataset with shape: %s", df_featured.shape)

    target_col = "charges" if "charges" in df_featured.columns else None
    if target_col is None:
        logger.warning("Target column 'charges' not found. Output will be features only.")

    # Split X/y
    X = df_featured.drop(columns=[target_col], errors="ignore") if target_col else df_featured.copy()
    y = df_featured[target_col] if target_col else None

    # Create and fit the preprocessor
    preprocessor = create_preprocessor(df_featured, target_col="charges")
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")

    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Saved preprocessor to %s", preprocessor_path)

    # Save fully preprocessed data (keep same format: CSV)
    feature_names = _get_feature_names(preprocessor)
    df_transformed = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=feature_names,
        index=df_featured.index,
    )

    if y is not None:
        df_transformed["charges"] = y.values

    df_transformed.to_csv(output_path, index=False)
    logger.info("Saved fully preprocessed data to %s", output_path)

    return df_transformed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering for insurance data.")
    parser.add_argument("--input", default="data/raw/insurance.csv")
    parser.add_argument("--output", default="data/processed/insurance_features_encoded.csv")
    parser.add_argument("--preprocessor", default="artifacts/preprocessor.joblib")

    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor)
