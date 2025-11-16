from pathlib import Path
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import mlflow
import mlflow.tensorflow

# MLflow setup
mlflow.set_experiment("energy_forecasting_lstm")
mlflow.tensorflow.autolog()

# Paths
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "src"))

results_models_dir = project_root / "results" / "models"
results_figures_dir = project_root / "results" / "figures"
results_models_dir.mkdir(parents=True, exist_ok=True)
results_figures_dir.mkdir(parents=True, exist_ok=True)

from utils.data_loader import EnergyConsumptionDataLoader  # noqa: E402


class LSTMForecaster:
    def __init__(
        self,
        input_weeks: int = 3,
        output_weeks: int = 1,
        stride: int = 3,
        lstm_units: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        recurrent_dropout: float = 0.2,
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 3e-4,
        clipnorm: float = 1.0,
        random_seed: int = 42,
    ):
        self.HOURS_PER_DAY = 24
        self.HOURS_PER_WEEK = 7 * self.HOURS_PER_DAY

        self.input_weeks = input_weeks
        self.output_weeks = output_weeks
        self.stride = stride

        self.input_window = self.input_weeks * self.HOURS_PER_WEEK   # 504
        self.output_window = self.output_weeks * self.HOURS_PER_WEEK # 168

        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm

        self.scaler_y: Optional[StandardScaler] = None
        self.model: Optional[tf.keras.Model] = None
        self.history = None

        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None

        tf.random.set_seed(random_seed)

    def _create_inout_sequences(
        self,
        series_2d: np.ndarray,
        input_window: int,
        output_window: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        N = len(series_2d)

        for start in range(0, N - input_window - output_window + 1, stride):
            end_input = start + input_window
            end_output = end_input + output_window

            X_seq = series_2d[start:end_input]
            y_seq = series_2d[end_input:end_output, 0]

            X.append(X_seq)
            y.append(y_seq)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def _build_model(self, n_features: int) -> tf.keras.Model:
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_window, n_features)))

        model.add(
            layers.LSTM(
                self.lstm_units,
                return_sequences=(self.num_lstm_layers > 1),
                recurrent_dropout=self.recurrent_dropout,
            )
        )

        for i in range(1, self.num_lstm_layers):
            return_sequences = (i < self.num_lstm_layers - 1)
            model.add(
                layers.LSTM(
                    self.lstm_units,
                    return_sequences=return_sequences,
                    recurrent_dropout=self.recurrent_dropout,
                )
            )

        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(self.lstm_units, activation="relu"))
        model.add(layers.Dense(self.output_window))

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.clipnorm,
        )

        model.compile(optimizer=optimizer, loss="mse")
        return model

    def prepare_data(
        self,
        train_series: pd.Series,
        valid_series: pd.Series,
    ) -> None:
        y_train = train_series.values.astype("float32").reshape(-1, 1)
        y_valid = valid_series.values.astype("float32").reshape(-1, 1)

        self.scaler_y = StandardScaler()
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_valid_scaled = self.scaler_y.transform(y_valid)

        X_train, Y_train = self._create_inout_sequences(
            y_train_scaled,
            input_window=self.input_window,
            output_window=self.output_window,
            stride=self.stride,
        )

        X_valid, Y_valid = self._create_inout_sequences(
            y_valid_scaled,
            input_window=self.input_window,
            output_window=self.output_window,
            stride=self.stride,
        )

        self.X_train, self.Y_train = X_train, Y_train
        self.X_valid, self.Y_valid = X_valid, Y_valid

        print("Prepared data for LSTM:")
        print("  X_train:", self.X_train.shape, "Y_train:", self.Y_train.shape)
        print("  X_valid:", self.X_valid.shape, "Y_valid:", self.Y_valid.shape)

    def fit(
        self,
        train_series: pd.Series,
        valid_series: pd.Series,
    ) -> None:
        self.prepare_data(train_series, valid_series)

        n_features = self.X_train.shape[2]

        self.model = self._build_model(n_features)
        self.model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        )

        self.history = self.model.fit(
            self.X_train,
            self.Y_train,
            validation_data=(self.X_valid, self.Y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

    def plot_loss(self) -> Path:
        if self.history is None:
            print("No training history found. Train the model first.")
            return None

        train_loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs_ran = range(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_ran, train_loss, label="Train loss")
        plt.plot(epochs_ran, val_loss, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.title("LSTM: Train vs Validation Loss")
        plt.legend()
        plt.grid(True)

        out = results_figures_dir / "lstm_loss.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out

    def predict_next_week_from_series(
        self,
        series: pd.Series,
    ) -> np.ndarray:
        if self.model is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        values = series.values.astype("float32").reshape(-1, 1)

        if len(values) < self.input_window:
            raise ValueError(
                f"Series too short. Need at least {self.input_window} points, "
                f"got {len(values)}."
            )

        context = values[-self.input_window:]

        context_scaled = self.scaler_y.transform(context)

        X_input = context_scaled[np.newaxis, :, :]

        y_pred_scaled = self.model.predict(X_input)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        y_pred_inverse = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        return y_pred_inverse

    def plot_next_week_forecast(
        self,
        series: pd.Series,
        label_prefix: str = "Valid",
    ) -> Path:
        y_pred = self.predict_next_week_from_series(series)

        context_values = series.values.astype("float32")
        past_week_true = context_values[-self.output_window:]

        plt.figure(figsize=(10, 5))
        plt.plot(
            range(-self.output_window, 0),
            past_week_true,
            label=f"{label_prefix} - last week (true)",
        )
        plt.plot(
            range(0, self.output_window),
            y_pred,
            label=f"{label_prefix} - next week (forecast)",
        )
        plt.axvline(0, color="black", linestyle="--", alpha=0.7)
        plt.xlabel("Hours relative to forecast start")
        plt.ylabel("PJME_MW")
        plt.title("LSTM Forecast: Last Week vs Next Week")
        plt.legend()
        plt.grid(True)

        out = results_figures_dir / "lstm_next_week_forecast.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        return out


def main():
    loader = EnergyConsumptionDataLoader(use_cuped=False)
    df = loader.load_data()
    train, valid, test = loader.train_valid_test_split()

    print("Train length:", len(train))
    print("Valid length:", len(valid))
    print("Test length:", len(test))
    print("Train head:")
    print(train.head())

    forecaster = LSTMForecaster(
        input_weeks=3,
        output_weeks=1,
        stride=3,
        lstm_units=64,
        num_lstm_layers=2,
        dropout=0.3,
        recurrent_dropout=0.2,
        batch_size=64,
        epochs=50,
        learning_rate=3e-4,
        clipnorm=1.0,
    )

    params = {
        "input_weeks": forecaster.input_weeks,
        "output_weeks": forecaster.output_weeks,
        "stride": forecaster.stride,
        "lstm_units": forecaster.lstm_units,
        "num_lstm_layers": forecaster.num_lstm_layers,
        "dropout": forecaster.dropout,
        "recurrent_dropout": forecaster.recurrent_dropout,
        "batch_size": forecaster.batch_size,
        "epochs": forecaster.epochs,
        "learning_rate": forecaster.learning_rate,
        "clipnorm": forecaster.clipnorm,
    }

    with mlflow.start_run(run_name="lstm_baseline"):
        mlflow.log_params(params)

        forecaster.fit(train_series=train, valid_series=valid)

        loss_path = forecaster.plot_loss()
        forecast_path = forecaster.plot_next_week_forecast(valid, label_prefix="Valid")

        if loss_path is not None:
            mlflow.log_artifact(str(loss_path), artifact_path="figures")
        if forecast_path is not None:
            mlflow.log_artifact(str(forecast_path), artifact_path="figures")

        model_save_path = results_models_dir / "lstm_model"
        forecaster.model.save(model_save_path)


if __name__ == "__main__":
    main()
