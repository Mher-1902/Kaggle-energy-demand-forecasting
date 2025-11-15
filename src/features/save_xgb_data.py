"""
Create XGBoost-ready features from CUPED PJME data
and save them into data/processed/.

Outputs:
    data/processed/xgb_features_full.csv
    data/processed/X_train.npy
    data/processed/y_train.npy
    data/processed/X_valid.npy
    data/processed/y_valid.npy
    data/processed/X_test.npy
    data/processed/y_test.npy
    data/processed/feature_columns.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make sure we can import utils.data_loader from project root
# ---------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]  # .../Energy-demand-forecasting
sys.path.append(str(project_root))

from utils.data_loader import EnergyConsumptionDataLoader


# ---------------------------------------------------------------------
# Simple feature engineering for XGBoost (no OOP)
# ---------------------------------------------------------------------
def feature_engineering_xgboost(
    df: pd.DataFrame,
    target_col: str = "PJME_MW_cuped",
    horizon: int = 1,
):
    """
    Feature engineering for XGBoost using CUPED (or original) PJME data.

    Args:
        df: DataFrame with datetime index or 'Datetime' column and target column.
        target_col: name of CUPED target column. If not found, falls back to 'PJME_MW'.
        horizon: how many steps ahead to predict (default 1 hour ahead).

    Returns:
        X: DataFrame of features
        y: Series of future target (shifted by -horizon)
        df_feat: full DataFrame with target + features + 'future_target'
    """
    df_feat = df.copy()

    # ---- Make sure we have a DatetimeIndex ----
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        if "Datetime" in df_feat.columns:
            df_feat = df_feat.set_index("Datetime")
        else:
            raise ValueError(
                "DataFrame must have DatetimeIndex or 'Datetime' column."
            )

    # ---- Choose target column (CUPED if exists, else original) ----
    if target_col not in df_feat.columns:
        if "PJME_MW" in df_feat.columns:
            target_col = "PJME_MW"
        else:
            raise ValueError(
                f"Target column '{target_col}' not found and 'PJME_MW' "
                f"also not in columns: {df_feat.columns.tolist()}"
            )

    y_series = df_feat[target_col]

    HOURS_PER_DAY = 24
    HOURS_PER_WEEK = 7 * HOURS_PER_DAY

    # ---- Lags and rolling windows ----
    lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]      # hours
    rolling_windows = [6, 12, 24, 48, 168, 336]      # hours

    # ---- Lag features ----
    for lag in lag_periods:
        df_feat[f"{target_col}_lag_{lag}"] = y_series.shift(lag)

    # ---- Rolling stats (using shift(1) to avoid look-ahead) ----
    for window in rolling_windows:
        base = f"{target_col}_roll_{window}"
        s_shift = y_series.shift(1)
        df_feat[f"{base}_mean"] = s_shift.rolling(window).mean()
        df_feat[f"{base}_std"] = s_shift.rolling(window).std()
        df_feat[f"{base}_min"] = s_shift.rolling(window).min()
        df_feat[f"{base}_max"] = s_shift.rolling(window).max()

    # ---- Differences vs recent lags ----
    df_feat[f"{target_col}_diff_1"] = y_series - y_series.shift(1)
    df_feat[f"{target_col}_diff_24"] = y_series - y_series.shift(24)
    df_feat[f"{target_col}_diff_168"] = y_series - y_series.shift(168)

    # ---- Time features ----
    idx = df_feat.index
    df_feat["hour"] = idx.hour
    df_feat["day_of_week"] = idx.dayofweek
    df_feat["day_of_month"] = idx.day
    df_feat["month"] = idx.month
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)

    # ---- Fourier seasonality (daily + weekly, k=1,2) ----
    t = np.arange(len(df_feat))

    daily_period = 24
    for k in [1, 2]:
        df_feat[f"fourier_daily_sin_{k}"] = np.sin(2 * np.pi * k * t / daily_period)
        df_feat[f"fourier_daily_cos_{k}"] = np.cos(2 * np.pi * k * t / daily_period)

    weekly_period = HOURS_PER_WEEK
    for k in [1, 2]:
        df_feat[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * t / weekly_period)
        df_feat[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * t / weekly_period)

    # ---- Build future target y(t + horizon) ----
    df_feat["future_target"] = df_feat[target_col].shift(-horizon)

    # ---- Drop rows with NaNs from lags/rolling/future_target ----
    df_feat = df_feat.dropna(subset=["future_target"])
    df_feat = df_feat.dropna()

    # ---- Build X, y ----
    drop_cols = {target_col, "future_target"}
    # also drop raw 'PJME_MW' if both exist and target_col is CUPED
    if "PJME_MW" in df_feat.columns and target_col != "PJME_MW":
        drop_cols.add("PJME_MW")

    feature_cols = [c for c in df_feat.columns if c not in drop_cols]

    X = df_feat[feature_cols]
    y = df_feat["future_target"]

    return X, y, df_feat, feature_cols


# ---------------------------------------------------------------------
# Main: load CUPED -> create features -> align with train/valid/test -> save
# ---------------------------------------------------------------------
def main():
    # 1) Load CUPED data using your loader
    loader = EnergyConsumptionDataLoader(use_cuped=True)
    df = loader.load_data()
    train, valid, test = loader.train_valid_test_split()

    print("Train length:", len(train))
    print("Valid length:", len(valid))
    print("Test length:", len(test))

    # 2) Feature engineering on FULL DF (then we'll split by index)
    X_full, y_full, df_feat, feature_cols = feature_engineering_xgboost(
        df,
        target_col="PJME_MW_cuped",  # falls back to PJME_MW if missing
        horizon=1,
    )

    # Make sure everything is sorted by time
    df_feat = df_feat.sort_index()
    X_full = X_full.sort_index()
    y_full = y_full.sort_index()

    # 3) Align with train/valid/test indices
    train_idx = train.index.intersection(X_full.index)
    valid_idx = valid.index.intersection(X_full.index)
    test_idx = test.index.intersection(X_full.index)

    X_train = X_full.loc[train_idx]
    y_train = y_full.loc[train_idx]

    X_valid = X_full.loc[valid_idx]
    y_valid = y_full.loc[valid_idx]

    X_test = X_full.loc[test_idx]
    y_test = y_full.loc[test_idx]

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 4) Save everything into data/processed/
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full feature DF (for inspection / notebooks)
    df_feat.to_csv(out_dir / "xgb_features_full.csv")
    print(f"[+] Saved full feature dataframe → {out_dir / 'xgb_features_full.csv'}")

    # NumPy arrays for fast XGBoost training
    np.save(out_dir / "X_train.npy", X_train.values)
    np.save(out_dir / "y_train.npy", y_train.values)
    np.save(out_dir / "X_valid.npy", X_valid.values)
    np.save(out_dir / "y_valid.npy", y_valid.values)
    np.save(out_dir / "X_test.npy", X_test.values)
    np.save(out_dir / "y_test.npy", y_test.values)

    print(f"[+] Saved X_train.npy, y_train.npy, X_valid.npy, y_valid.npy, X_test.npy, y_test.npy")

    # Save feature column names
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=4)
    print(f"[+] Saved feature columns → {out_dir / 'feature_columns.json'}")

    print("\n=== DONE: XGBoost data saved to data/processed/ ===")


if __name__ == "__main__":
    main()
