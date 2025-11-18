import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that creates domain, statistical,
    and interaction features from an insurance dataframe.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def fit(self, X, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        if self.verbose:
            print("[FeatureEngineering] Creating domain, statistical, and interaction features...")

        df = self.create_domain_features(df)
        df = self.create_statistical_features(df)
        df = self.create_interaction_features(df)

        return df

    # --- migrated notebook logic ---

    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain / semantic features that don't use the target."""
        df_feat = df.copy()

        # 1. BMI categories (safe - no target information)
        if 'bmi' in df_feat.columns:
            df_feat['bmi_category'] = pd.cut(
                df_feat['bmi'],
                bins=[0, 18.5, 25, 30, 35, 40, 100],
                labels=['Underweight', 'Normal', 'Overweight',
                        'Obese_I', 'Obese_II', 'Obese_III']
            )
            df_feat['bmi_category_num'] = df_feat['bmi_category'].map({
                'Underweight': 1,
                'Normal': 0,
                'Overweight': 2,
                'Obese_I': 3,
                'Obese_II': 4,
                'Obese_III': 5
            })

        # 2. Age Group (safe)
        if 'age' in df_feat.columns:
            df_feat['age_group'] = pd.cut(
                df_feat['age'],
                bins=[17, 25, 35, 45, 55, 65],
                labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elder']
            )

        # 3. Family Size
        if 'children' in df_feat.columns:
            df_feat['family_size'] = df_feat['children'].apply(
                lambda x: 'Single' if x == 0
                else 'small' if x <= 2
                else 'large'
            )

        # 4. Create health status score
        def calculate_health_status(row):
            """Calculate a simple health score based only on input features."""
            score = 100
            if 'bmi' in row and pd.notnull(row['bmi']):
                if row['bmi'] < 18.5 or row['bmi'] > 30:
                    score -= 20
                elif row['bmi'] > 25:
                    score -= 10
            if 'smoker' in row and row.get('smoker') == 'yes':
                score -= 30
            if 'age' in row and pd.notnull(row['age']):
                if row['age'] > 50:
                    score -= 10
                elif row['age'] > 40:
                    score -= 5
            return score

        if set(['age', 'bmi', 'smoker']).intersection(df_feat.columns):
            df_feat['health_score'] = df_feat.apply(calculate_health_status, axis=1)

        # 5. Age - BMI interaction (safe)
        if set(['age', 'bmi']).issubset(df_feat.columns):
            df_feat['age_bmi_interaction'] = df_feat['age'] * df_feat['bmi'] / 100

        if self.verbose:
            print("✅ Created domain features")

        return df_feat

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical / transformed features."""
        df_feat = df.copy()

        if self.verbose:
            print("Creating statistical features...")

        # Log transformations
        if 'bmi' in df_feat.columns:
            df_feat['log_bmi'] = np.log1p(df_feat['bmi'].clip(lower=0))
        if 'charges' in df_feat.columns:
            df_feat['log_charges'] = np.log1p(df_feat['charges'].clip(lower=0))

        # Square root transformations
        if 'age' in df_feat.columns:
            df_feat['sqrt_age'] = np.sqrt(df_feat['age'].clip(lower=0))
        if 'bmi' in df_feat.columns:
            df_feat['sqrt_bmi'] = np.sqrt(df_feat['bmi'].clip(lower=0))

        # Polynomial features
        if 'age' in df_feat.columns:
            df_feat['age_squared'] = df_feat['age'] ** 2
        if 'bmi' in df_feat.columns:
            df_feat['bmi_squared'] = df_feat['bmi'] ** 2

        # Binning
        if 'age' in df_feat.columns:
            df_feat['age_decade'] = (df_feat['age'] // 10) * 10

        if self.verbose:
            print("✅ Created statistical features")

        return df_feat

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        df_feat = df.copy()

        if self.verbose:
            print("Creating interaction features...")

        # Age-BMI interactions
        if set(['age', 'bmi']).issubset(df_feat.columns):
            df_feat['age_bmi_ratio'] = df_feat['age'] / (df_feat['bmi'] + 1)

        # Smoker interactions (binary encoding)
        if 'smoker' in df_feat.columns:
            smoker_binary = df_feat['smoker'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
            if 'age' in df_feat.columns:
                df_feat['smoker_age'] = smoker_binary * df_feat['age']
            if 'bmi' in df_feat.columns:
                df_feat['smoker_bmi'] = smoker_binary * df_feat['bmi']

        # Children interactions
        if set(['children', 'age']).issubset(df_feat.columns):
            df_feat['children_per_age'] = df_feat['children'] / (df_feat['age'] + 1)

        # Sex interactions
        if 'sex' in df_feat.columns:
            male_binary = (df_feat['sex'] == 'male').astype(int)
            if 'bmi' in df_feat.columns:
                df_feat['male_bmi'] = male_binary * df_feat['bmi']
            if 'age' in df_feat.columns:
                df_feat['male_age'] = male_binary * df_feat['age']

        if self.verbose:
            print("✅ Created interaction features")

        return df_feat

    def create_safe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full safe feature engineering pipeline and return dataframe."""
        if self.verbose:
            print("\n🚀 SAFE FEATURE ENGINEERING PIPELINE")

        df_feat = df.copy()
        df_feat = self.create_domain_features(df_feat)
        df_feat = self.create_statistical_features(df_feat)
        df_feat = self.create_interaction_features(df_feat)

        if self.verbose:
            print("\n📊 FEATURE ENGINEERING SUMMARY")
            print(f"Original features: {df.shape[1]}")
            print(f"New features created: {df_feat.shape[1] - df.shape[1]}")
            print(f"Total features: {df_feat.shape[1]}")

        return df_feat
