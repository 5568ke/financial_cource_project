from .kmeans import KMeansCluster
from .dbscan import DBSCANCluster
# from clustering.agglomerative import AgglomerativeCluster
# from clustering.faiss import FaissCluster

class ClusterFactory:
    """
    Factory to create clustering model instances based on specified algorithm.
    """
    def __init__(self, config):
        self.config = config

    def get_cluster_model_instance(self):
        """
        Creates an instance of the specified clustering algorithm based on the configuration.
        """
        algorithm = self.config.CLUSTERING_ALGORITHM
        if algorithm == 'kmeans':
            return KMeansCluster(**self.config.KMEANS_PARAMS)
        elif algorithm == 'dbscan':
            return DBSCANCluster(**self.config.DBSCAN_PARAMS)
        # elif algorithm == 'agglomerative':
        #     return AgglomerativeCluster(**self.config.AGGLOMERATIVE_PARAMS)
        # elif algorithm == 'faiss':
        #     return FaissCluster(**self.config.FAISS_PARAMS)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
