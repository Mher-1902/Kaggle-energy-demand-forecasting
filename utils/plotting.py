import sys
from pathlib import Path
project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))
print("Added to path:", project_root / "src")
from utils.data_loader import EnergyConsumptionDataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
loader = EnergyConsumptionDataLoader("../config/config.yaml")
df = loader.load_data()
y_train, y_valid, y_test = loader.train_valid_test_split()
plt.figure(figsize=(15,4))
plt.plot(df, color='tab:blue')
plt.title("PJME Energy Demand - Full History")
plt.xlabel("Date")
plt.ylabel("MW")
plt.show()
plt.figure(figsize=(15,4))
plt.plot(y_train, label="Train")
plt.plot(y_valid, label="Valid")
plt.plot(y_test, label="Test")
plt.title("Train / Validation / Test Split")
plt.legend()
plt.show()
plt.figure(figsize=(15,4))
plt.plot(y_train[-2000:], label="Train (last part)")
plt.plot(y_valid, label="Validation")
plt.title("Last 2000 hours of Train + Validation Window")
plt.legend()
plt.show()
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Daily seasonality = 24 hours
stl_daily = STL(df['PJME_MW'], period=24, robust=True)
res_daily = stl_daily.fit()

fig, axes = plt.subplots(3, 1, figsize=(12, 7))

axes[0].plot(df.index, res_daily.trend, color='red')
axes[0].set_title("Trend Component (Daily Seasonality)")

axes[1].plot(df.index, res_daily.seasonal, color='blue')
axes[1].set_title("Seasonal Component (Daily, Period=24)")

axes[2].plot(df.index, res_daily.resid, color='black')
axes[2].set_title("Residual Component")

plt.tight_layout()
plt.show()
# Weekly seasonality = 168 hours (7 days)
stl_weekly = STL(df['PJME_MW'], period=168, robust=True)
res_weekly = stl_weekly.fit()

fig, axes = plt.subplots(3, 1, figsize=(12, 7))

axes[0].plot(df.index, res_weekly.trend, color='red')
axes[0].set_title("Trend Component (Weekly Seasonality)")

axes[1].plot(df.index, res_weekly.seasonal, color='blue')
axes[1].set_title("Seasonal Component (Weekly, Period=168)")

axes[2].plot(df.index, res_weekly.resid, color='black')
axes[2].set_title("Residual Component")

plt.tight_layout()
plt.show()

last_date = df.index.max()
start_date = last_date - pd.DateOffset(years=2)
df_last_year = df.loc[start_date:last_date].copy()

def month_to_season(month):
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"

df_last_year["season"] = df_last_year.index.month.map(month_to_season)
df_last_year["date"] = df_last_year.index.date
df_last_year["hour"] = df_last_year.index.hour
import plotly.express as px

fig = px.line(
    df_last_year,
    x=df_last_year.index,          
    y="PJME_MW",
    color="season",                
    hover_data={
        "season": True,
        "date": True,
        "hour": True,
        "PJME_MW": ":.0f"
    },
    labels={
        "PJME_MW": "Demand (MW)",
        "x": "Datetime"
    },
    title="PJME Hourly Demand – Last Year by Season"
)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(df["PJME_MW"], ax=ax[0], lags=100)
plot_pacf(df["PJME_MW"], ax=ax[1], lags=50)
plt.show()
# =====================================================================
# Why did we choose SARIMA(3, d, q) × (1, D, 1, 24)?
# =====================================================================
# 1. ACF (Autocorrelation Function)
#    - Shows strong repeating peaks at lags 24, 48, 72, 96...
#    - This is clear evidence of DAILY seasonality (24 hours).
#    - Strong ACF spike at lag 24 → include a seasonal MA term (Q=1).
#
# 2. PACF (Partial Autocorrelation Function)
#    - Sharp drop after lag 3: lag 1, 2, 3 are significant, then it cuts off.
#    - This is the classic signature of an AR(p) process.
#    - Therefore: p = 3.
#    - PACF also shows a spike at lag 24 → include seasonal AR term (P=1).
#
# 3. Seasonal period
#    - Data is hourly.
#    - Daily seasonality = 24 hours → s = 24.
#
# 4. Summary of components:
#    - Non-seasonal AR(p):      p = 3   (PACF cutoff)
#    - Non-seasonal MA(q):      q = 0 or 1 (ACF does not sharply cut)
#    - Seasonal AR(P):          P = 1   (PACF spike at 24)
#    - Seasonal MA(Q):          Q = 1   (ACF spike at 24)
#    - Seasonal period (s):     24
#
# 5. Final reasoning:
#    - The data has strong autoregression at short lags → AR(3).
#    - It has strong daily autocorrelation → seasonal AR and MA.
#    - Therefore we choose: SARIMA(3, d, q) × (1, D, 1, 24).
#
# Note:
#    - d and D (differencing terms) depend on whether the series is 
#      stationary. We'll find these using ADF test or auto_arima.
# =====================================================================
plt.figure(figsize=(8,4))
sns.histplot(df["PJME_MW"], bins=50, kde=True)
plt.title("Distribution of Energy Demand")
plt.show()

#Since this data is skwed CUPED will help reduce variance
df_hour = df.copy()
df_hour["hour"] = df_hour.index.hour

plt.figure(figsize=(12,6))
sns.boxplot(x="hour", y="PJME_MW", data=df_hour)
plt.title("Demand by Hour of Day")
plt.show()
df_dow = df.copy()
df_dow["dow"] = df_dow.index.dayofweek

plt.figure(figsize=(12,6))
sns.boxplot(x="dow", y="PJME_MW", data=df_dow)
plt.title("Demand by Day of Week")
plt.show()
