from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from clustering_evaluator import evaluate_clustering

X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# KMeans
print("\n--- KMeans ---")
evaluate_clustering(KMeans(n_clusters=4, random_state=42), X)

# DBSCAN
print("\n--- DBSCAN ---")
evaluate_clustering(DBSCAN(eps=0.5), X)
