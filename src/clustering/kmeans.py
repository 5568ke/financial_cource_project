from sklearn.cluster import KMeans as SKLearnKMeans
from cluster_base import ClusterBase

class KMeansCluster(ClusterBase):
    def __init__(self, n_clusters=8, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = SKLearnKMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, stocks, date):
        features_list = [stock.get_features_for_clustering(date) for stock in stocks]
        data = np.array(features_list)
        self.model.fit(data)

    def predict(self, stocks, date):
        features_list = [stock.get_features_for_clustering(date) for stock in stocks]
        data = np.array(features_list)
        return self.model.predict(data)

    def get_cluster_centers(self):
        return self.model.cluster_centers_
