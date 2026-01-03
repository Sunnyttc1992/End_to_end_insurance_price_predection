import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data."""
    logger.info("Creating new features")
    
    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # Calculate Age buckets (non-linear risk)
    df_featured["age_group"] = pd.cut(
        df_featured["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    logger.info("Created 'age_group' feature")
    
    # BMI categories (medical standard)
    df_featured["bmi_category"] = pd.cut(
        df_featured["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )
    logger.info("Created 'bmi-categories' feature")
    
    # Smoker risk flag
    df_featured["is_smoker"] = (df_featured["smoker"] == "yes").astype(int)
    logger.info("Created 'Smoker - risk' feature")

    # High-risk BMI indicator
    df_featured["high_bmi"] = (df_featured["bmi"] >= 30).astype(int)
    logger.info("Created 'age_smoker_interaction' feature")

    # Age × smoker interaction
    df_featured["age_smoker_interaction"] = df_featured["age"] * df_featured["is_smoker"]
    logger.info("Created 'high-bmi' indicator")

    # BMI × smoker interaction
    df_featured["bmi_smoker_interaction"] = df_featured["bmi"] * df_featured["is_smoker"]
    logger.info("Created BMI × smoker interaction")

    # Family size
    df_featured["family_size"] = df_featured["children"] + 1
    logger.info("Created 'family_size' feature")

    # Large family flag
    df_featured["large_family"] = (df_featured["family_size"] >= 4).astype(int)
    logger.info("Created 'large_family' flag")
    
    # Do NOT one-hot encode categorical variables here; let the preprocessor handle it
    return df_featured

def create_preprocessor(df_featured):
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessor pipeline")
    # Categorical = object or category
    categorical_features = df_featured.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numerical = ints + floats
    numerical_features = df_featured.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full feature engineering pipeline."""
    # Load cleaned data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Create features
    df_featured = create_features(df)
    logger.info(f"Created featured dataset with shape: {df_featured.shape}")
    
    # Create and fit the preprocessor
    X = df_featured.drop(columns=['charges'], errors='ignore')  # Features only
    preprocessor = create_preprocessor(X)
    y = df_featured['charges'] if 'charges' in df_featured.columns else None  # Target column (if available)
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")
    
    # Save the preprocessor
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")
    
    # Save fully preprocessed data
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['charges'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")
    
    return df_transformed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering for housing data.')
    parser.add_argument("--input", default="data/processed/clean_insurance.csv")
    parser.add_argument("--output", default="data/processed/featured.csv")
    parser.add_argument("--preprocessor", default="artifacts/preprocessor.pkl")
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output, args.preprocessor)
