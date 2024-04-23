from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

# Import the ClusterBase class
from .cluster_base import ClusterBase


class DBSCANCluster(ClusterBase):
    def __init__(self, alpha=0.1, non_feature_columns=None):
        super().__init__(non_feature_columns)
        self.alpha = alpha

    def fit(self, feature_data):
        models_dfs = []
        all_data_frames = []

        for month, data in tqdm(feature_data.groupby("DATE"), desc="Training DBSCAN clusters"):
            feature_subset = self.filter_feature_data(data)
            if len(feature_subset) < 2:
                print(f"Skipping {month} due to insufficient data.")
                continue

            distances = self.compute_distances(feature_subset, int(round(np.log(len(feature_subset))) + 1))
            eps = np.percentile(distances, self.alpha * 100)

            db_model = DBSCAN(eps=eps, min_samples=int(round(np.log(len(feature_subset)))), metric='euclidean')
            db_model.fit(feature_subset)

            annotated_data = self.annotate_data(data, db_model.labels_, distances, eps)
            all_data_frames.append(annotated_data)
            num_clusters = len(set(db_model.labels_)) - (1 if -1 in db_model.labels_ else 0)
            models_dfs.append({'DATE': month, 'n_clusters': num_clusters})

        annotated_df = pd.concat(all_data_frames, ignore_index=True)
        models_df = pd.DataFrame(models_dfs)

        return models_df, annotated_df