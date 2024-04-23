import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class DBSCANCluster:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def distance_to_nearest_neighbors(self, data, k, alpha):
        """ Calculate the epsilon distance from the nearest neighbors. """
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        distances = np.sort(distances[:, -1])
        eps = np.percentile(distances, alpha * 100)
        return eps

    def fit(self, feature_data):
        """ Perform DBSCAN clustering and return annotated data with cluster labels. """
        models_dfs = []
        all_data_frames = []

        for month, data in tqdm(feature_data.groupby("DATE"), desc="Training DBSCAN clusters"):
            # Remove 'DATE' and 'permno' columns to focus on numerical data for clustering
            feature_subset = data.drop(['DATE', 'permno'], axis=1)
            if len(feature_subset) < 2:
                print(f"Skipping {month} due to insufficient data.")
                continue

            # Estimate the minimum number of samples as the natural logarithm of the number of data points
            min_samples = int(round(np.log(len(feature_subset))))
            # Calculate the epsilon distance for DBSCAN
            eps = self.distance_to_nearest_neighbors(feature_subset, k=min_samples + 1, alpha=self.alpha)
            db_model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            db_model.fit(feature_subset)

            # Annotate data with cluster labels, including permno for identification post-clustering
            data['cluster'] = db_model.labels_

            # Append annotated data
            all_data_frames.append(data)
            num_clusters = len(set(db_model.labels_)) - (1 if -1 in db_model.labels_ else 0)
            models_dfs.append({'DATE': month, 'n_clusters': num_clusters})

        # Concatenate all annotated dataframes for output
        annotated_df = pd.concat(all_data_frames, ignore_index=True)
        models_df = pd.DataFrame(models_dfs)

        return models_df, annotated_df


