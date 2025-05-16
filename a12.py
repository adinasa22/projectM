import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os, logging, warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
import shap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class AdvancedEDA:
    def __init__(self, df: pd.DataFrame, target: str = None, report_dir="eda_reports"):
        self.df = df.copy()
        self.target = target
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

        if self.target:
            self.problem_type = 'classification' if self.df[self.target].nunique() <= 10 else 'regression'
        else:
            self.problem_type = None

        logging.info(f"Initialized Advanced EDA with target: {self.target}, problem_type: {self.problem_type}")

    def general_info(self):
        logging.info("Generating general info and basic stats.")
        print("Shape:", self.df.shape)
        print("Types:\n", self.df.dtypes)
        print("Missing Values:\n", self.df.isnull().sum())
        self.df.describe(include='all').to_csv(f"{self.report_dir}/summary_statistics.csv")

    def encode_data(self):
        logging.info("Encoding categorical features.")
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))

    def time_series_check(self):
        logging.info("Checking for time series data.")
        datetime_cols = self.df.select_dtypes(include='datetime').columns
        if not datetime_cols.empty:
            for col in datetime_cols:
                self.df[col] = pd.to_datetime(self.df[col])
                self.df.set_index(col, inplace=True)
                self.df.sort_index(inplace=True)
                plt.figure(figsize=(10, 4))
                self.df[self.target].resample('M').mean().plot(title=f"{self.target} over time")
                plt.savefig(f"{self.report_dir}/time_series_{col}.png")
                plt.close()

    def correlation_and_clustering(self):
        logging.info("Generating correlation heatmap and dendrogram.")
        num_df = self.df.select_dtypes(include=np.number)
        corr = num_df.corr()

        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Matrix")
        plt.savefig(f"{self.report_dir}/correlation_matrix.png")
        plt.close()

        # Clustering
        plt.figure(figsize=(10, 6))
        Z = hierarchy.linkage(corr, 'ward')
        hierarchy.dendrogram(Z, labels=corr.columns, leaf_rotation=90)
        plt.title("Hierarchical Feature Clustering")
        plt.savefig(f"{self.report_dir}/feature_dendrogram.png")
        plt.close()

    def pca_visualization(self):
        logging.info("Running PCA for visualization.")
        num_df = self.df.select_dtypes(include=np.number).drop(columns=[self.target], errors='ignore')
        scaled = StandardScaler().fit_transform(num_df)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        if self.target:
            pca_df["target"] = self.df[self.target]
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target")
        else:
            sns.scatterplot(data=pca_df, x="PC1", y="PC2")
        plt.title("PCA Projection")
        plt.savefig(f"{self.report_dir}/pca_projection.png")
        plt.close()

    def class_imbalance_analysis(self):
        if self.problem_type != 'classification':
            return
        logging.info("Analyzing class imbalance.")
        target_counts = self.df[self.target].value_counts(normalize=True)
        target_counts.plot(kind='bar', title="Class Imbalance")
        plt.savefig(f"{self.report_dir}/class_imbalance.png")
        plt.close()

    def feature_importance_shap(self):
        logging.info("Running SHAP-based feature importance.")
        X = self.df.drop(columns=[self.target], errors='ignore')
        y = self.df[self.target]

        model = RandomForestClassifier() if self.problem_type == "classification" else RandomForestRegressor()
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        plt.figure()
        if self.problem_type == "classification":
            shap.summary_plot(shap_values[1], X, show=False)
        else:
            shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(f"{self.report_dir}/shap_summary.png")
        plt.close()

    def univariate_multivariate_analysis(self):
        logging.info("Running univariate and multivariate visualizations.")
        for col in self.df.select_dtypes(include=np.number).columns[:6]:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.savefig(f"{self.report_dir}/hist_{col}.png")
            plt.close()

            if self.target and col != self.target:
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=self.df[self.target], y=self.df[col]) if self.problem_type == 'classification' else sns.scatterplot(x=self.df[col], y=self.df[self.target])
                plt.title(f"{col} vs {self.target}")
                plt.savefig(f"{self.report_dir}/interaction_{col}.png")
                plt.close()

    def run_all(self):
        self.general_info()
        self.encode_data()
        self.time_series_check()
        self.correlation_and_clustering()
        self.pca_visualization()
        self.class_imbalance_analysis()
        self.univariate_multivariate_analysis()
        if self.target:
            self.feature_importance_shap()
        logging.info("Advanced EDA complete!")


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("your_dataset.csv") # Replace with your dataset
    eda = AdvancedEDA(df, target="target") # Replace 'target' with your target
    eda.run_all()
