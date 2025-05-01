import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs  # For example data
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_dbscan_hyperparameters(X, eps_range, min_samples_range):
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}

    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)

            # Ignore single cluster or all noise
            if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in labels):
                continue

            try:
                score = silhouette_score(X, labels)
                results.append((eps, min_samples, score))

                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}

                logger.info(f"Evaluated eps={eps}, min_samples={min_samples}, silhouette={score:.4f}")
            except Exception as e:
                logger.warning(f"Failed for eps={eps}, min_samples={min_samples}: {e}")

    return best_params, results


def plot_heatmap(results):
    data = np.array(results)
    heatmap_data = {}
    for eps, min_samples, score in results:
        heatmap_data[(eps, min_samples)] = score

    eps_values = sorted(set(row[0] for row in results))
    min_samples_values = sorted(set(row[1] for row in results))

    heatmap = np.zeros((len(min_samples_values), len(eps_values)))

    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            heatmap[i, j] = heatmap_data.get((eps, min_samples), np.nan)

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap, xticklabels=eps_values, yticklabels=min_samples_values, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Epsilon")
    plt.ylabel("Min Samples")
    plt.title("Silhouette Score Heatmap for DBSCAN")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example dataset
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
    X = StandardScaler().fit_transform(X)

    # Define parameter grid
    eps_range = np.arange(0.1, 1.5, 0.1)
    min_samples_range = range(2, 10)

    # Optimize
    best_params, results = optimize_dbscan_hyperparameters(X, eps_range, min_samples_range)

    print(f"\nBest Parameters Found: {best_params}")

    # Visualize
    plot_heatmap(results)
