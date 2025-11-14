# import sys
# from pathlib import Path
# project_root = Path().resolve().parent
# sys.path.append(str(project_root / "src"))
# print("Added to path:", project_root / "src")
# from utils.data_loader import EnergyConsumptionDataLoader
# loader = EnergyConsumptionDataLoader("../config/config.yaml")
# df = loader.load_data()
# y_train, y_valid, y_test = loader.train_valid_test_split()
import pandas as pd

class CupedTransformer:
    def __init__(self,lag_hours: int = 168,center: bool = True):
        self.lag_hours = lag_hours
        self.center = center
        self.theta_ = None
        self.x_mean = None
        self.is_fitted_ = False


    def _build_lag(self,y: pd.Series):
        return y.shift(self.lag_hours)

    def fit(self,y_train: pd.Series):
        x_train = self._build_lag(y_train)

        data = pd.concat([y_train,x_train],axis=1).dropna()

        y = data.iloc[:, 0]
        x = data.iloc[:, 1]

        x_mean = x.mean()
        y_mean = y.mean()

        cov_yx = ((y - y_mean) * (x - x_mean)).mean()
        var_x = ((x - x_mean) ** 2).mean()

        if var_x == 0:
            raise ValueError("Variance of X is zero; CUPED cannot be applied.")

        theta = cov_yx / var_x

        self.theta_ = theta
        self.x_mean_ = x_mean if self.center else 0.0
        self.is_fitted_ = True
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")

        x = self._build_lag(y)

        if self.center:
            y_cuped = y - self.theta_ * (x - self.x_mean_)
        else:
            y_cuped = y - self.theta_ * x

        return y_cuped

    def inverse_transform(self, y_cuped: pd.Series, y_original_for_lag: pd.Series) -> pd.Series:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before inverse_transform().")

        x = self._build_lag(y_original_for_lag)

        if self.center:
            y_rec = y_cuped + self.theta_ * (x - self.x_mean_)
        else:
            y_rec = y_cuped + self.theta_ * x

        return y_rec



