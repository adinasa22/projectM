import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Generate sample data (replace with your own dataset)
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Step 1: Find optimal epsilon using k-distance plot
k = 4 # Common practice: min_samples = dimensionality + 1
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort distances to find elbow
k_distances = np.sort(distances[:, k-1])

# Plot k-distance graph
plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title("K-Distance Graph")
plt.xlabel("Data Points sorted by distance")
plt.ylabel(f"{k}th Nearest Neighbor Distance")
plt.grid(True)
plt.show()

# Step 2: Choose eps from plot (example: around the 'elbow')
# You might manually select based on the plot, or loop through candidate values:
best_eps = None
best_score = -1
best_model = None

for eps in np.arange(0.1, 1.0, 0.05):
    model = DBSCAN(eps=eps, min_samples=k)
    labels = model.fit_predict(X)
    # Ignore noise points in silhouette score
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_eps = eps
            best_model = model

print(f"Best eps: {best_eps}, Best silhouette score: {best_score}")
