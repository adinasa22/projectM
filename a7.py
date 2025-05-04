def optimize_dbscan_hyperparameters(X, eps_range, min_samples_range):
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}

    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            eps = round(eps, 2)
            min_samples = round(min_samples, 2)
            print(eps, min_samples)


import pandas as pd
import numpy as np
eps_range = np.arange(0.1, 1.5, 0.1)
min_samples_range = range(2, 10)
X = pd.DataFrame()

optimize_dbscan_hyperparameters(X, eps_range, min_samples_range)
