import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from itertools import product


def find_optimal_dbscan_params_by_outliers(X, target_n_outliers, eps_range=(0.1, 2.0), min_samples_range=(3, 10),
                                           step=0.1):
    eps_values = np.arange(eps_range[0], eps_range[1], step)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)

    heatmap_data = np.zeros((len(min_samples_values), len(eps_values)))

    best_params = None
    best_diff = float('inf')
    best_labels = None
    best_coords = None

    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            n_outliers = np.sum(labels == -1)
            heatmap_data[i, j] = n_outliers
            diff = abs(n_outliers - target_n_outliers)
            if diff < best_diff:
                best_diff = diff
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels
                best_coords = (i, j)

    return best_params, best_diff, best_labels, heatmap_data, eps_values, min_samples_values, best_coords


def plot_outlier_heatmap(heatmap_data, eps_values, min_samples_values, best_coords, target_n_outliers):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        cbar_kws={'label': 'Number of Outliers'}
    )
    ax.set_xticks(np.arange(len(eps_values)) + 0.5)
    ax.set_yticks(np.arange(len(min_samples_values)) + 0.5)
    ax.set_xticklabels(np.round(eps_values, 2))
    ax.set_yticklabels(min_samples_values)
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.title(f"DBSCAN Outlier Counts (Target: {target_n_outliers})")

    # Highlight best parameters
    ax.add_patch(plt.Rectangle((best_coords[1], best_coords[0]), 1, 1, fill=False, edgecolor='lime', lw=3))
    plt.show()


def plot_clusters(X, labels):
    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for label, col in zip(unique_labels, colors):
        if label == -1:
            col = (1, 0, 0, 1)  # red for outliers
            marker = 'x'
        else:
            marker = 'o'

        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6, linestyle='None')

    plt.title("DBSCAN Clustering Results (Outliers in Red)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.6, random_state=42)
    target_n_outliers = 50  # Specify the number of desired outliers

    params, diff, labels, heatmap_data, eps_values, min_samples_values, best_coords = find_optimal_dbscan_params_by_outliers(
        X, target_n_outliers=target_n_outliers)

    print("Best Parameters:", params)
    print("Actual Number of Outliers:", np.sum(labels == -1))

    plot_outlier_heatmap(heatmap_data, eps_values, min_samples_values, best_coords, target_n_outliers)
    plot_clusters(X, labels)