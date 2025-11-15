from pathlib import Path
import sys

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.data_loader import EnergyConsumptionDataLoader


def run_sarima(
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    train_len_limit=20_000,
):
    
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

    if len(train) > train_len_limit:
        y_train = train[-train_len_limit:]
        print(f"Using last {train_len_limit} points of train for SARIMA.")
    else:
        y_train = train
        print(f"Using full train of length {len(train)} for SARIMA.")

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

    n_valid = len(valid)
    n_test = len(test)
    total_steps = n_valid + n_test

    forecast_res = res.get_forecast(steps=total_steps)
    forecast_values = forecast_res.predicted_mean  

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


    results_dir = PROJECT_ROOT / "results" / "forecasts"
    results_dir.mkdir(parents=True, exist_ok=True)

    valid_path = results_dir / "sarima_valid.csv"
    test_path = results_dir / "sarima_test.csv"

    forecast_valid.to_csv(valid_path, header=True)
    forecast_test.to_csv(test_path, header=True)

    print("Saved validation forecast to:", valid_path)
    print("Saved test forecast to:", test_path)

    return res, forecast_valid, forecast_test



if __name__ == "__main__":
    run_sarima()
