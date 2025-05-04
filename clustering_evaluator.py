import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clustering(model, X, plot=True):
    """
    Fits and evaluates a clustering model using internal metrics.
    Optionally plots the clusters in 2D using PCA.

    Parameters:
    - model: scikit-learn compatible clustering model
    - X: input features (numpy array or DataFrame)
    - plot: bool, whether to show 2D PCA cluster plot

    Returns:
    - dict: Evaluation metrics
    """
    try:
        labels = model.fit_predict(X)
    except AttributeError:
        model.fit(X)
        labels = model.labels_

    unique_clusters = np.unique(labels)
    print(f"[INFO] Number of clusters found: {len(unique_clusters)}")

    scores = {}
    if len(unique_clusters) > 1:
        scores['Silhouette Score'] = silhouette_score(X, labels)
        scores['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
        scores['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
    else:
        scores['Silhouette Score'] = None
        scores['Calinski-Harabasz Index'] = None
        scores['Davies-Bouldin Index'] = None

    print("[INFO] Evaluation Metrics:")
    for metric, score in scores.items():
        if score is not None:
            print(f" {metric}: {score:.4f}")
        else:
            print(f" {metric}: Not applicable (only one cluster)")

    if plot and X.shape[1] > 1:
        _plot_clusters(X, labels)

    return scores

def _plot_clusters(X, labels):
    """
    Helper function to plot clusters in 2D using PCA.
    """
    X_reduced = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=30)
    plt.title("Cluster Visualization (PCA Reduced)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()
