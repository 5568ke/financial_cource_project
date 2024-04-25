import pandas as pd
import numpy as np
from tqdm import tqdm
from .cluster_base import ClusterBase
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import AgglomerativeClustering
class AgglomerativeCluster(ClusterBase):
    def __init__(self, alpha=0.3, linkage='average', non_feature_columns=None):
        super().__init__(non_feature_columns)
        self.alpha = alpha
        self.linkage = linkage

    def fit(self, feature_data):
        models_dfs = []
        all_data_frames = []

        for month, data in tqdm(feature_data.groupby("DATE"), desc="Training Agglomerative clusters"):
            feature_subset = self.filter_feature_data(data)
            if len(feature_subset) < 2:
                print(f"Skipping {month} due to insufficient data.")
                continue

            # Fit AgglomerativeClustering model
            distances = self.compute_distances(feature_subset,k=2)
            eps = np.percentile(distances, self.alpha * 100)
            agg_model = AgglomerativeClustering(n_clusters=None, distance_threshold=eps, linkage=self.linkage)
            agg_model.fit(feature_subset)

            # Annotate original data with cluster labels
            annotated_data = self.annotate_data(data, agg_model.labels_, distances, eps)

            # Append annotated data
            all_data_frames.append(annotated_data)

            # Record model details for each month
            models_dfs.append({'DATE': month, 'n_clusters': agg_model.n_clusters_})

        # Concatenate all annotated dataframes for output
        annotated_df = pd.concat(all_data_frames, ignore_index=True)
        models_df = pd.DataFrame(models_dfs)

        return models_df, annotated_df
