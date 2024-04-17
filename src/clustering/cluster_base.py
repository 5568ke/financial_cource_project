from abc import ABC, abstractmethod

class ClusterBase(ABC):
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters

    @abstractmethod
    def fit(self, X):
        """Fit the model to X."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        pass

