"""
LSTM time series forecaster for PJME energy consumption.

- Input:  last 3 weeks (504 hours)
- Output: next 1 week (168 hours)
- Uses stride to reduce number of training samples
- OOP-style class: LSTMForecaster

Place this file in e.g.:
    src/models/lstm_model.py
"""

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

# Make sure project root is on the path so we can import utils.data_loader
project_root = Path("..").resolve()
sys.path.append(str(project_root))

from utils.data_loader import EnergyConsumptionDataLoader


class LSTMForecaster:
    """
    LSTM forecaster for univariate time series (PJME_MW).

    - input_weeks:  history length in weeks (3 => 3 * 168 = 504 hours)
    - output_weeks: forecast horizon in weeks (1 => 168 hours)
    - stride:       window step in hours (3 means we take every 3rd hour as a new window)
    """

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

        # For debugging / inspection
        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.Y_valid = None

        tf.random.set_seed(random_seed)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _create_inout_sequences(
        self,
        series_2d: np.ndarray,
        input_window: int,
        output_window: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a scaled series into (X, y) windows.

        series_2d: shape (N, n_features). Here n_features = 1 (just PJME_MW).
        Returns:
            X: (num_samples, input_window, n_features)
            y: (num_samples, output_window)
        """
        X, y = [], []
        N = len(series_2d)

        for start in range(0, N - input_window - output_window + 1, stride):
            end_input = start + input_window
            end_output = end_input + output_window

            X_seq = series_2d[start:end_input]           # (input_window, 1)
            y_seq = series_2d[end_input:end_output, 0]   # (output_window,)

            X.append(X_seq)
            y.append(y_seq)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def _build_model(self, n_features: int) -> tf.keras.Model:
        """
        Build the LSTM model.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_window, n_features)))

        # First LSTM layer
        model.add(
            layers.LSTM(
                self.lstm_units,
                return_sequences=(self.num_lstm_layers > 1),
                recurrent_dropout=self.recurrent_dropout,
            )
        )

        # Additional LSTM layers if num_lstm_layers > 1
        for i in range(1, self.num_lstm_layers):
            # On the last LSTM layer we don't return sequences
            return_sequences = (i < self.num_lstm_layers - 1)
            model.add(
                layers.LSTM(
                    self.lstm_units,
                    return_sequences=return_sequences,
                    recurrent_dropout=self.recurrent_dropout,
                )
            )

        # Dropout + Dense layers
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(self.lstm_units, activation="relu"))
        model.add(layers.Dense(self.output_window))  # 168 outputs: next 168 hours

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.clipnorm,
        )

        model.compile(optimizer=optimizer, loss="mse")
        return model

    # -----------------------------
    # Public API
    # -----------------------------

    def prepare_data(
        self,
        train_series: pd.Series,
        valid_series: pd.Series,
    ) -> None:
        """
        Scale train/valid and create (X, y) windows.
        """
        # Convert to 2D arrays (N, 1)
        y_train = train_series.values.astype("float32").reshape(-1, 1)
        y_valid = valid_series.values.astype("float32").reshape(-1, 1)

        # Fit scaler on train only
        self.scaler_y = StandardScaler()
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_valid_scaled = self.scaler_y.transform(y_valid)

        # Create windows
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
        """
        Full training pipeline:
        - scale data
        - create windows
        - build model
        - train with early stopping
        """
        # 1. Prepare data
        self.prepare_data(train_series, valid_series)

        n_features = self.X_train.shape[2]  # should be 1 for univariate

        # 2. Build model
        self.model = self._build_model(n_features)
        self.model.summary()

        # 3. Callbacks for overfitting control
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

        # 4. Train
        self.history = self.model.fit(
            self.X_train,
            self.Y_train,
            validation_data=(self.X_valid, self.Y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

    def plot_loss(self) -> None:
        """
        Plot train vs validation loss to inspect overfitting.
        """
        if self.history is None:
            print("No training history found. Train the model first.")
            return

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
        plt.show()

    def predict_next_week_from_series(
        self,
        series: pd.Series,
    ) -> np.ndarray:
        """
        Take the LAST `input_window` hours from the given series,
        and predict the NEXT `output_window` hours (1 week).

        Returns:
            y_pred_inverse: np.ndarray of shape (output_window,)
            in original PJME_MW scale.
        """
        if self.model is None or self.scaler_y is None:
            raise RuntimeError("Model is not trained. Call fit() first.")

        values = series.values.astype("float32").reshape(-1, 1)

        if len(values) < self.input_window:
            raise ValueError(
                f"Series too short. Need at least {self.input_window} points, "
                f"got {len(values)}."
            )

        # Take last input_window as context
        context = values[-self.input_window:]  # (input_window, 1)

        # Scale using the same scaler
        context_scaled = self.scaler_y.transform(context)

        # Model expects shape (1, input_window, n_features)
        X_input = context_scaled[np.newaxis, :, :]  # (1, input_window, 1)

        y_pred_scaled = self.model.predict(X_input)
        # y_pred_scaled shape: (1, output_window)
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)  # (output_window, 1)

        # Inverse transform back to MW
        y_pred_inverse = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        return y_pred_inverse

    def plot_next_week_forecast(
        self,
        series: pd.Series,
        label_prefix: str = "Valid",
    ) -> None:
        """
        Plot last week of true data vs next week forecast.

        We:
        - use the last input_window points as context
        - plot:
            - the last output_window hours of the context (true past week)
            - the predicted next output_window hours
        """
        y_pred = self.predict_next_week_from_series(series)

        context_values = series.values.astype("float32")
        past_week_true = context_values[-self.output_window:]  # last real week

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
        plt.show()


# ---------------------------------------------------------
# Example script usage
# ---------------------------------------------------------

def main():
    # 1. Load data
    loader = EnergyConsumptionDataLoader(use_cuped=False)
    df = loader.load_data()
    train, valid, test = loader.train_valid_test_split()

    print("Train length:", len(train))
    print("Valid length:", len(valid))
    print("Test length:", len(test))
    print("Train head:")
    print(train.head())

    # 2. Create forecaster
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

    # 3. Train on train, validate on valid
    forecaster.fit(train_series=train, valid_series=valid)

    # 4. Plot train vs validation loss
    forecaster.plot_loss()

    # 5. Example: forecast next week from end of validation segment
    forecaster.plot_next_week_forecast(valid, label_prefix="Valid")


if __name__ == "__main__":
    main()
