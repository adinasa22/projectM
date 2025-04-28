import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO)

class DriftDetector:
    def __init__(self, baseline_data: pd.DataFrame):
        """
        baseline_data: DataFrame of training data (before deployment)
        """
        self.baseline = baseline_data
        self.feature_columns = baseline_data.columns.tolist()

    def detect_drift(self, new_data: pd.DataFrame, threshold=0.05):
        """
        Compare new incoming data with baseline data.
        threshold: p-value threshold for statistical significance
        """
        drift_report = {}

        for col in self.feature_columns:
            if col not in new_data.columns:
                logging.warning(f"Column {col} not found in new data. Skipping...")
                continue

            baseline_col = self.baseline[col].dropna()
            new_col = new_data[col].dropna()

            # Only numeric features for now
            if np.issubdtype(baseline_col.dtype, np.number) and np.issubdtype(new_col.dtype, np.number):
                stat, p_value = ks_2samp(baseline_col, new_col)
                drifted = p_value < threshold
                drift_report[col] = {
                    "p_value": p_value,
                    "drift_detected": drifted
                }
            else:
                logging.info(f"Skipping non-numeric column {col}")

        return drift_report

    def summarize_drift(self, drift_report):
        """
        Print a summary of drift detection.
        """
        for feature, report in drift_report.items():
            status = "DRIFT DETECTED" if report["drift_detected"] else "No drift"
            print(f"Feature: {feature}, p-value: {report['p_value']:.4f}, Status: {status}")


# 1. Load your baseline (training) data
print('Loading baseline data...')
baseline_df = pd.read_csv("bank_comma.csv")

# 2. New data collected in production
print('Loading new data...')
new_data_df = pd.read_csv("bank_comma.csv")

# 3. Create detector
print('Creating drift detector...')
detector = DriftDetector(baseline_df)

# 4. Run detection
print('Running drift detection...')
drift_report = detector.detect_drift(new_data_df)

# 5. Summarize results
print('Summarizing results...')
detector.summarize_drift(drift_report)