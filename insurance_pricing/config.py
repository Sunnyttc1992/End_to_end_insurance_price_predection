from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "insurance.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # if you keep an explicit validation split

TARGET_COL = "charges"

# Original features from the classic insurance dataset
NUM_FEATURES = ["age", "bmi", "children"]
CAT_FEATURES = ["sex", "smoker", "region"]
