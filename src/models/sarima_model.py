from pathlib import Path
import sys

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------

# project root: .../Energy-demand-forecasting
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.data_loader import EnergyConsumptionDataLoader


# ---------------------------------------------------------------------
# Main SARIMA runner
# ---------------------------------------------------------------------

def run_sarima(
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    train_len_limit=20_000,
):
    """
    Fit a SARIMA model on RAW PJME_MW using EnergyConsumptionDataLoader,
    then forecast on validation + test and save forecasts.

    Parameters
    ----------
    order : tuple
        Non-seasonal (p, d, q).
    seasonal_order : tuple
        Seasonal (P, D, Q, s), here s=24 for daily seasonality.
    train_len_limit : int
        If the training series is very long, use only the last
        `train_len_limit` points to speed up fitting.

    Returns
    -------
    res : SARIMAXResults
        Fitted SARIMAX results object.
    forecast_valid : pd.Series
        Forecast for the validation period.
    forecast_test : pd.Series
        Forecast for the test period.
    """

    # 1) Load data (RAW, no CUPED)
    loader = EnergyConsumptionDataLoader(
        config_path=str(PROJECT_ROOT / "config" / "config.yaml"),
        use_cuped=False,
    )
    df = loader.load_data()
    train, valid, test = loader.train_valid_test_split()

    print("Full data shape:", df.shape)
    print("Train length:", len(train))
    print("Valid length:", len(valid))
    print("Test  length:", len(test))

    # 2) Optionally limit training length to speed up
    if len(train) > train_len_limit:
        y_train = train[-train_len_limit:]
        print(f"Using last {train_len_limit} points of train for SARIMA.")
    else:
        y_train = train
        print(f"Using full train of length {len(train)} for SARIMA.")

    # 3) Fit SARIMA
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    print("Fitting SARIMA...")
    res = model.fit(disp=False)
    print(res.summary())

    # 4) Forecast for validation + test in one go
    n_valid = len(valid)
    n_test = len(test)
    total_steps = n_valid + n_test

    forecast_res = res.get_forecast(steps=total_steps)
    forecast_values = forecast_res.predicted_mean  # pandas Series

    # 5) Split forecast into valid + test and align indices
    forecast_valid = pd.Series(
        forecast_values.iloc[:n_valid].values,
        index=valid.index,
        name="sarima_valid",
    )

    forecast_test = pd.Series(
        forecast_values.iloc[n_valid:].values,
        index=test.index,
        name="sarima_test",
    )

    # 6) Compute metrics
    val_mse = mean_squared_error(valid, forecast_valid)
    val_mae = mean_absolute_error(valid, forecast_valid)

    test_mse = mean_squared_error(test, forecast_test)
    test_mae = mean_absolute_error(test, forecast_test)

    print("\n================ SARIMA RESULTS ================")
    print("Order:", order, "Seasonal order:", seasonal_order)
    print("Train used:", len(y_train))
    print("\n--- Validation ---")
    print("MSE:", val_mse)
    print("MAE:", val_mae)
    print("\n--- Test ---")
    print("MSE:", test_mse)
    print("MAE:", test_mae)
    print("================================================\n")

    # 7) Save forecasts
    results_dir = PROJECT_ROOT / "results" / "forecasts"
    results_dir.mkdir(parents=True, exist_ok=True)

    valid_path = results_dir / "sarima_valid.csv"
    test_path = results_dir / "sarima_test.csv"

    forecast_valid.to_csv(valid_path, header=True)
    forecast_test.to_csv(test_path, header=True)

    print("Saved validation forecast to:", valid_path)
    print("Saved test forecast to:", test_path)

    return res, forecast_valid, forecast_test


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Default config: SARIMA(1,0,1) × (1,0,1,24) on RAW PJME_MW
    run_sarima()
