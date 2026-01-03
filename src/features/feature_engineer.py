import pandas as pd


def engineer_features(df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Apply feature engineering to an insurance DataFrame and return the transformed DataFrame.
    If `save_path` is provided, the result is saved as CSV to that path.
    """
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    df["is_smoker"] = (df["smoker"] == "yes").astype(int)
    df["high_bmi"] = (df["bmi"] >= 30).astype(int)

    df["age_smoker_interaction"] = df["age"] * df["is_smoker"]
    df["bmi_smoker_interaction"] = df["bmi"] * df["is_smoker"]

    df["family_size"] = df["children"] + 1
    df["large_family"] = (df["family_size"] >= 4).astype(int)

    region_freq = df["region"].value_counts(normalize=True)
    df["region_freq"] = df["region"].map(region_freq)
    df["region_smoker"] = df["region"].astype(str) + "_" + df["smoker"].astype(str)

    df["is_adult"] = (df["age"] >= 18).astype(int)
    df["prime_risk_age"] = ((df["age"] >= 40) & (df["age"] <= 60)).astype(int)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
