import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ComplexEDA:
    def __init__(self, df: pd.DataFrame, target_col: str = None, report_dir: str = "eda_reports"):
        self.df = df.copy()
        self.target_col = target_col
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        logging.info("EDA initialized.")

    def overview(self):
        logging.info("Generating data overview...")
        print("Shape:", self.df.shape)
        print("\nData Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isnull().sum())
        print("\nDescriptive Stats:\n", self.df.describe(include='all'))

    def encode_categoricals(self):
        logging.info("Encoding categorical columns for analysis.")
        le = LabelEncoder()
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = le.fit_transform(self.df[col])

    def missing_values_analysis(self):
        logging.info("Analyzing missing values...")
        missing = self.df.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            missing.plot(kind='barh', figsize=(10, 6), title="Missing Value Percentage")
            plt.tight_layout()
            plt.savefig(f"{self.report_dir}/missing_values.png")
            plt.close()

    def outlier_detection_boxplots(self):
        logging.info("Generating outlier boxplots...")
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot - {col}")
            plt.savefig(f"{self.report_dir}/boxplot_{col}.png")
            plt.close()

    def correlation_heatmap(self):
        logging.info("Generating correlation heatmap...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=False, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.report_dir}/correlation_heatmap.png")
        plt.close()

    def target_distribution(self):
        if self.target_col:
            logging.info(f"Analyzing target variable: {self.target_col}")
            plt.figure(figsize=(6, 4))
            if self.df[self.target_col].nunique() <= 10:
                sns.countplot(x=self.target_col, data=self.df)
            else:
                sns.histplot(self.df[self.target_col], kde=True)
            plt.title(f"Distribution of Target: {self.target_col}")
            plt.tight_layout()
            plt.savefig(f"{self.report_dir}/target_distribution.png")
            plt.close()

    def pairplot_interactions(self):
        logging.info("Generating pairplot for first few numerical features...")
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()[:5]
        sns.pairplot(self.df[num_cols])
        plt.savefig(f"{self.report_dir}/pairplot.png")
        plt.close()

    def chi_squared_test(self):
        if not self.target_col:
            return
        logging.info("Running chi-squared test for categorical features.")
        categorical_cols = self.df.select_dtypes(include='object').columns
        results = []
        for col in categorical_cols:
            if col == self.target_col: continue
            contingency = pd.crosstab(self.df[col], self.df[self.target_col])
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            results.append((col, p))
        significant = [(col, p) for col, p in results if p < 0.05]
        logging.info(f"Significant categorical features: {significant}")

    def anova_test(self):
        if not self.target_col:
            return
        logging.info("Running ANOVA for numerical features against target.")
        results = []
        for col in self.df.select_dtypes(include=np.number).columns:
            if col == self.target_col: continue
            try:
                groups = [v[col].dropna() for k, v in self.df.groupby(self.target_col)]
                f_stat, p = stats.f_oneway(*groups)
                results.append((col, p))
            except Exception:
                continue
        significant = [(col, p) for col, p in results if p < 0.05]
        logging.info(f"Significant numerical features: {significant}")

    def calculate_vif(self):
        logging.info("Calculating VIF for multicollinearity...")
        X = self.df.select_dtypes(include=np.number).drop(columns=[self.target_col], errors='ignore')
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        logging.info("\n" + str(vif_data))

    def run_full_eda(self):
        self.overview()
        self.encode_categoricals()
        self.missing_values_analysis()
        self.outlier_detection_boxplots()
        self.correlation_heatmap()
        self.target_distribution()
        self.pairplot_interactions()
        self.chi_squared_test()
        self.anova_test()
        self.calculate_vif()
        logging.info("Full EDA completed.")


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("bank_comma.csv")  # Replace with your dataset
    eda = ComplexEDA(df, target_col="y")  # Replace 'target' with your target column
    eda.run_full_eda()


