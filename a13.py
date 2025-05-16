import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest

# --------------- Step 1: Generate Sample Time Series Data ---------------
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
ts = pd.Series(np.random.normal(loc=0, scale=1, size=len(date_rng)), index=date_rng)
ts.iloc[10] += 10 # Inject outliers
ts.iloc[40] -= 8

# --------------- Step 2: Stationarity Check and Auto Differencing ---------------
def check_stationarity(ts, alpha=0.05):
    result = adfuller(ts.dropna())
    print("\n--- ADF Test ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value:.4f}")
    return result[1] <= alpha

def ensure_stationarity(ts):
    if check_stationarity(ts):
        print("✅ Series is stationary. Proceeding...")
        return ts, 0
    else:
        print("⚠️ Series is not stationary. Applying differencing...")
        ts_diff = ts.diff().dropna()
        if check_stationarity(ts_diff):
            print("✅ Stationarity achieved after differencing.")
            return ts_diff, 1
        else:
            print("⚠️ Even after differencing, the series is not stationary.")
            return ts_diff, 1

ts_stationary, diff_order = ensure_stationarity(ts)

# --------------- Step 3: Outlier Detection Methods ---------------
# 1. Rolling Z-Score
def detect_rolling_zscore_outliers(ts, window=7, threshold=3):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    z_scores = (ts - rolling_mean) / rolling_std
    outliers = np.abs(z_scores) > threshold
    return outliers, z_scores

# 2. Seasonal Decomposition
def detect_seasonal_decompose_outliers(ts, model='additive', freq=7, threshold=3):
    decomposition = seasonal_decompose(ts, model=model, period=freq)
    residual = decomposition.resid.dropna()
    z_scores = (residual - residual.mean()) / residual.std()
    outliers = np.abs(z_scores) > threshold
    return outliers, residual

# 3. Isolation Forest
def detect_isolation_forest_outliers(ts, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    ts_reshaped = ts.values.reshape(-1, 1)
    preds = clf.fit_predict(ts_reshaped)
    outliers = preds == -1
    return pd.Series(outliers, index=ts.index)

outliers_zscore, z_scores = detect_rolling_zscore_outliers(ts_stationary)
outliers_decomp, residual = detect_seasonal_decompose_outliers(ts_stationary)
outliers_iforest = detect_isolation_forest_outliers(ts_stationary)

# --------------- Step 4: Plot Results ---------------
plt.figure(figsize=(15, 5))
plt.plot(ts_stationary, label='(Stationary) Time Series')
plt.scatter(ts_stationary[outliers_zscore].index, ts_stationary[outliers_zscore], color='red', label='Z-Score Outliers')
plt.scatter(ts_stationary[outliers_decomp].index, ts_stationary[outliers_decomp], color='orange', marker='x', label='Decompose Outliers')
plt.scatter(ts_stationary[outliers_iforest].index, ts_stationary[outliers_iforest], color='purple', marker='s', label='Isolation Forest Outliers')
plt.title("Outlier Detection on Stationary Time Series")
plt.legend()
plt.show()
