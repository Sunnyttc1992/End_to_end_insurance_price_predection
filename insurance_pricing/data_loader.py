import pandas as pd
from .config import RAW_DATA

def load_insurance_data() -> pd.DataFrame:
    """
    Load raw insurance data from CSV.
    """
    df = pd.read_csv(RAW_DATA)
    return df