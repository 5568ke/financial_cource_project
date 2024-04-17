from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm

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

    def fit(self, pca_result_df):
        models_dfs = []
        clean_dataframes = []
        outlier_dataframes = []
        for month, data in tqdm(pca_result_df.groupby("DATE"), desc="train_km_clusters"):            
            pca_data = data.drop('DATE', axis=1)
            if len(pca_data) < 2:
                print(f"Skipping {month} due to insufficient data.")
                continue
            self.model.fit(pca_data)

            # Calculate distances to centroids
            distances = self.model.transform(pca_data)
            min_distances = np.min(distances, axis=1)
            threshold = np.percentile(min_distances, 95)

            # Identify outliers and clean data
            clean_data = data[min_distances <= threshold]
            outlier_data = data[min_distances > threshold]

            # Save cluster labels and outlier identification
            clean_data['km_cluster'] = self.model.labels_[min_distances <= threshold]
            outlier_data.loc['km_cluster'] = -1

            # Append to lists
            clean_dataframes.append(clean_data)
            outlier_dataframes.append(outlier_data)

            # Save model details
            models_dfs.append({'DATE': month, 'n_clusters': self.n_clusters})

        # Concatenate dataframes
        clean_df = pd.concat(clean_dataframes, ignore_index=True)
        outlier_df = pd.concat(outlier_dataframes, ignore_index=True)
        models_df = pd.DataFrame(models_dfs)

        return models_df, clean_df, outlier_df

    def predict(self, data):
        return self.model.predict(data)
