import pandas as pd
import numpy as np
import yaml
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EnergyConsumptionDataLoader:
    def __init__(
        self,
        config_path: str = "/home/rog/Documents/Energy-demand-forecasting/config/config.yaml",
        use_cuped: bool = False,
        processed_path: str | None = None,):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config["data"]

        self.use_cuped = use_cuped
        self.processed_path = (
            Path(processed_path)
            if processed_path is not None
            else PROJECT_ROOT / "data/processed/pjme_cuped.csv"

        )

        self.df = None
        self.train = None
        self.valid = None
        self.test = None
        self.target_col_used = None  

    def load_data(self):
        if self.use_cuped:
            raw_path = self.processed_path
            base_target = self.data_config["target_col"]
            target_col = f"{base_target}_cuped"
        else:
            # Use original raw dataset
            raw_path = self.data_config["raw_path"]
            target_col = self.data_config["target_col"]

        try:
            df = pd.read_csv(raw_path)
        except Exception as e:
            raise RuntimeError(f"Couldn't load dataset: {e}")

        if "Datetime" not in df.columns:
            raise ValueError("CSV file must contain Datetime column")

        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        self.df = df
        self.target_col_used = target_col
        return df

    def train_valid_test_split(self):
        if self.df is None:
            raise RuntimeError("Call load_data() first before split")

        if self.target_col_used is not None:
            target_col = self.target_col_used
        else:
            base_target = self.data_config["target_col"]
            target_col = f"{base_target}_cuped" if self.use_cuped else base_target

        y = self.df[target_col]

        valid_hours = self.data_config["valid_days"] * 24
        test_hours = self.data_config["test_days"] * 24

        if len(y) <= valid_hours + test_hours:
            raise ValueError("Time series too short for this split")

        self.train = y.iloc[: -(valid_hours + test_hours)]
        self.valid = y.iloc[-(valid_hours + test_hours) : -test_hours]
        self.test = y.iloc[-test_hours:]

        return self.train, self.valid, self.test
