from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ClusterBase(ABC):
    """
    Abstract base class for clustering algorithms.
    """

    def __init__(self, non_feature_columns=None):
        if non_feature_columns is None:
            non_feature_columns = ['DATE', 'permno']
        self.non_feature_columns = non_feature_columns

    def filter_feature_data(self, data):
        """ Remove non-numeric or identifier columns that are not features for clustering. """
        return data.drop(columns=self.non_feature_columns, errors='ignore')

    def compute_distances(self, data, k):
        """ Calculate the distance to the k-th nearest neighbor for each point in the dataset. """
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        return np.sort(distances[:, -1])

    @abstractmethod
    def fit(self, data):
        """ Fit the clustering model to the data. """
        pass

    def annotate_data(self, data, labels, min_distances, threshold):
        """ Annotate the data with cluster labels and outlier detection. """
        data['cluster'] = labels
        data['is_outlier'] = min_distances > threshold
        return data