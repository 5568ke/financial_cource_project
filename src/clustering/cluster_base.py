from abc import ABC, abstractmethod

class ClusterBase(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

