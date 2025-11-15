from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.data_loader import EnergyConsumptionDataLoader  


def make_prophet_df(y: pd.Series) -> pd.DataFrame:

    df = y.reset_index()
    df.columns = ["ds", "y"]
    return df


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": mae, "rmse": rmse}


def main(
    use_cuped: bool = False,
    processed_path: str | None = None,
    results_path: str | Path = "results/prophet_results.csv",
):

    if processed_path is None:
        processed_path = project_root / "data" / "processed" / "pjme_cuped.csv"

    loader = EnergyConsumptionDataLoader(
        use_cuped=use_cuped,
        processed_path=str(processed_path),
    )

    df = loader.load_data()
    train, valid, test = loader.train_valid_test_split()

    print("Full shape:", df.shape)
    print("Train length:", len(train))
    print("Valid length:", len(valid))
    print("Test length:", len(test))

    y_train = train
    y_valid = valid
    y_test = test

    y_train_valid = pd.concat([y_train, y_valid])
    train_valid_df = make_prophet_df(y_train_valid)
    test_df = make_prophet_df(y_test)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(train_valid_df)

    n_test = len(y_test)
    future = m.make_future_dataframe(
        periods=n_test,
        freq="H", 
        include_history=True,
    )

    forecast = m.predict(future)

    forecast_test = forecast.set_index("ds").loc[test_df["ds"]]
    y_pred_test = forecast_test["yhat"]

    metrics = evaluate_forecast(y_test.values, y_pred_test.values)

    print("Prophet on test set:")
    print(f"  MAE  = {metrics['mae']:.3f}")
    print(f"  RMSE = {metrics['rmse']:.3f}")

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model": "prophet",
        "use_cuped": use_cuped,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_train": len(y_train_valid),
        "n_test": len(y_test),
    }

    if results_path.exists():
        results_df = pd.read_csv(results_path)
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([row])

    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main(
        use_cuped=False,
        processed_path="/home/rog/Documents/Energy-demand-forecasting/data/processed/pjme_cuped.csv",
    )
