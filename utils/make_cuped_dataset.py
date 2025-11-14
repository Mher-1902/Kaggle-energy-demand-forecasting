from pathlib import Path
import sys
import pandas as pd
import yaml

# Add project root (NOT /src) to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
print("Added to path:", project_root)

from src.cuped import CupedTransformer


CONFIG_PATH = Path("/home/rog/Documents/Energy-demand-forecasting/config/config.yaml")
PROCESSED_PATH = Path("data/processed/pjme_cuped.csv")
LAG_HOURS = 168  # 1 week lag for covariate


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    raw_path = Path(data_cfg["raw_path"])
    target_col = data_cfg["target_col"]
    valid_days = int(data_cfg["valid_days"])
    test_days = int(data_cfg["test_days"])

    # 1) Load raw
    df = pd.read_csv(raw_path, parse_dates=["Datetime"])
    df = df.set_index("Datetime").sort_index()

    y = df[target_col]

    # 2) Compute splits by last N days
    n = len(df)
    valid_len = valid_days * 24
    test_len = test_days * 24

    test_start = n - test_len
    valid_start = n - test_len - valid_len

    # Train = everything before validation+test
    train_df = df.iloc[:valid_start]
    y_train = train_df[target_col]

    # 3) Fit CUPED only on train
    cuped = CupedTransformer(lag_hours=LAG_HOURS)
    cuped.fit(y_train)

    # 4) Transform full series with same theta & mean
    y_cuped_full = cuped.transform(y)
    df[f"{target_col}_cuped"] = y_cuped_full

    # 5) Drop initial NaNs from lag
    df = df.dropna(subset=[f"{target_col}_cuped"])

    # 6) Save processed dataset
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH)

    print("Saved:", PROCESSED_PATH)
    print("Shape:", df.shape)
    print("theta:", cuped.theta_, "x_mean:", cuped.x_mean_)
    print("valid_start idx:", valid_start, "test_start idx:", test_start)


if __name__ == "__main__":
    main()
