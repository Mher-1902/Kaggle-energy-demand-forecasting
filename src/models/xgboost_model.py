import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"
results_models_dir = project_root / "results" / "models"
results_models_dir.mkdir(parents=True, exist_ok=True)


def load_xgb_data():
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy").ravel()
    X_valid = np.load(processed_dir / "X_valid.npy")
    y_valid = np.load(processed_dir / "y_valid.npy").ravel()
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy").ravel()
    with open(processed_dir / "feature_columns.json", "r") as f:
        feature_names = json.load(f)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names


def train_xgboost(X_train, y_train, X_valid, y_valid, feature_names):
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
    params = {
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "seed": 42,
    }
    evals = [(dtrain, "train"), (dvalid, "valid")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return bst


def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names):
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    y_pred_train = model.predict(dtrain)
    y_pred_valid = model.predict(dvalid)
    y_pred_test = model.predict(dtest)

    rmse_tr, mae_tr = metrics(y_train, y_pred_train)
    rmse_va, mae_va = metrics(y_valid, y_pred_valid)
    rmse_te, mae_te = metrics(y_test, y_pred_test)

    print(rmse_tr, mae_tr)
    print(rmse_va, mae_va)
    print(rmse_te, mae_te)
def plot_test_week(model, X_test, y_test, feature_names):
    import pandas as pd
    import matplotlib.pyplot as plt

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    y_pred = model.predict(dtest)

    df_feat = pd.read_csv(processed_dir / "xgb_features_full.csv",
                          index_col=0, parse_dates=True)

    test_idx = df_feat.index[-len(y_test):]

    horizon = 168
    idx = test_idx[-horizon:]
    true_vals = y_test[-horizon:]
    pred_vals = y_pred[-horizon:]

    plt.figure(figsize=(10, 5))
    plt.plot(idx, true_vals, label="true")
    plt.plot(idx, pred_vals, label="pred")
    plt.legend()
    plt.tight_layout()

    out = project_root / "results" / "figures" / "xgb_test_last_week.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)


def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names = load_xgb_data()
    model = train_xgboost(X_train, y_train, X_valid, y_valid, feature_names)
    evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names)
    model_path = results_models_dir / "xgb_model.json"
    model.save_model(model_path)
    plot_test_week(model, X_test, y_test, feature_names)



if __name__ == "__main__":
    main()
