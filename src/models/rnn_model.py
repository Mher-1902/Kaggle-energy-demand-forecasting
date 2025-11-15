from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.data_loader import EnergyConsumptionDataLoader  # noqa: E402


def make_supervised(
    y: np.ndarray,
    window: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    
    X, target = [], []
    for i in range(len(y) - window):
        X.append(y[i : i + window])
        target.append(y[i + window])
    X = np.array(X)
    target = np.array(target)
    return X[..., np.newaxis], target


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": mae, "rmse": rmse}


def build_lstm_model(window: int = 24) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, input_shape=(window, 1)))
    model.add(Dense(1))

    model.compile(
        loss="mse",
        optimizer="adam",
    )
    return model


def main(
    use_cuped: bool = False,
    processed_path: str | None = None,
    window: int = 24,
    results_path: str | Path = "results/rnn_results.csv",
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

    y_train_valid = np.concatenate([y_train.values, y_valid.values])
    y_test_arr = y_test.values

    X_train, y_train_sup = make_supervised(y_train_valid, window=window)

    y_all = np.concatenate([y_train_valid, y_test_arr])
    X_all, y_all_sup = make_supervised(y_all, window=window)

    n_train_sup = len(y_train_sup)
    X_test = X_all[n_train_sup:]
    y_test_sup = y_all_sup[n_train_sup:]


    model = build_lstm_model(window=window)
    model.summary()

    es = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train_sup,
        epochs=50,
        batch_size=64,
        verbose=1,
        callbacks=[es],
    )


    y_pred_test = model.predict(X_test).flatten()

    metrics = evaluate_forecast(y_test_sup, y_pred_test)

    print("RNN (LSTM) on test sup set:")
    print(f"  MAE  = {metrics['mae']:.3f}")
    print(f"  RMSE = {metrics['rmse']:.3f}")

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model": "rnn_lstm",
        "use_cuped": use_cuped,
        "window": window,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "n_train_sup": len(y_train_sup),
        "n_test_sup": len(y_test_sup),
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
        window=24,
    )
