import pandas as pd
from config import Config

class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        path = Config.RAW_DATA_DIR + self.filename
        return pd.read_csv(path)

    def preprocess_data(self, data):
        return data

