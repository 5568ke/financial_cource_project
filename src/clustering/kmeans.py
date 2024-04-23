from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearnex import patch_sklearn
patch_sklearn()

class KMeansCluster:
    def __init__(self, n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, init=self.init,
                            n_init=self.n_init, max_iter=self.max_iter,
                            random_state=self.random_state)

    def fit(self, feature_data):
        """ Perform KMeans clustering and return annotated data with clusters and outlier flags. """
        models_dfs = []
        all_data_frames = []

        for month, data in tqdm(feature_data.groupby("DATE"), desc="Training KMeans clusters"):
            # Remove 'DATE' and 'permno' columns before clustering
            feature_subset = data.drop(['DATE', 'permno'], axis=1)
            if len(feature_subset) < 2:
                print(f"Skipping {month} due to insufficient data.")
                continue

            self.model.fit(feature_subset)

            # Calculate distances to centroids for each point
            distances = self.model.transform(feature_subset)
            min_distances = np.min(distances, axis=1)
            threshold = np.percentile(min_distances, 95)

            # Annotate original data with cluster labels and determine outliers
            data['cluster'] = self.model.labels_
            data['is_outlier'] = min_distances > threshold  # Marking points as outliers if their distance to the nearest cluster center is above the threshold

            # Append annotated data
            all_data_frames.append(data)

            # Record model details for each month
            models_dfs.append({'DATE': month, 'n_clusters': self.n_clusters})

        # Concatenate all annotated dataframes for output
        annotated_df = pd.concat(all_data_frames, ignore_index=True)
        models_df = pd.DataFrame(models_dfs)

        return models_df, annotated_df



