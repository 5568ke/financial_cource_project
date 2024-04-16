import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from tqdm import tqdm
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from concurrent.futures import ProcessPoolExecutor


class DataProcessor:
    NAN_THRESHOLD = 0.15
    WINDOW = 48  

    def __init__(self, filepath, filepath_parquet, min_year, max_year):
        self.filepath = filepath
        self.filepath_parquet = filepath_parquet
        self.min_year = min_year
        self.max_year = max_year

    def load_data(self):
        print("Step: Start loading data")

        if not os.path.exists(self.filepath_parquet):
            self.dataframe = pd.read_csv(self.filepath)
            self.dataframe.to_parquet(self.filepath_parquet, index=False, compression='snappy')
        else:
            self.dataframe = pd.read_parquet(self.filepath_parquet)

        print("Finish: Data loading")
        return self.dataframe

    def preprocess_data(self):
        print("Step: Start preprocessing data")

        self.dataframe['DATE'] = pd.to_datetime(self.dataframe['DATE'], format='%Y%m%d')
        self.dataframe = self.dataframe[(self.dataframe['DATE'].dt.year >= self.min_year) &
                                        (self.dataframe['DATE'].dt.year <= self.max_year)]
        self.dataframe = self.dataframe.sort_values("DATE")

        print("Finish: Preprocessing data")
        return self.dataframe

    # def sanitize_and_feature_engineer(self):
    #     print("Step: Start sanitization and feature engineering")
    #
    #     self.dataframe = self.dataframe.groupby('permno').filter(lambda x: len(x) >= self.WINDOW)
    #     self.calculate_momentum()
    #     self.interpolate_with_median()
    #     self.remove_high_nan_features()
    #
    #     print("Finish: Sanitization and feature engineering")
    #     return self.dataframe
    #
    def sanitize_and_feature_engineer(self):
        filepath_feature_engineered = self.filepath_parquet.replace('.parquet', '_features.parquet')

        if os.path.exists(filepath_feature_engineered):
            print("Step: Loading preprocessed and feature engineered data")
            self.dataframe = pd.read_parquet(filepath_feature_engineered)
            print("Finish: Loading preprocessed and feature engineered data")
        else:
            print("Step: Start sanitization and feature engineering")
            self.dataframe = self.dataframe.groupby('permno').filter(lambda x: len(x) >= self.WINDOW)
            self.calculate_momentum()
            self.interpolate_with_median()
            self.remove_high_nan_features()
            
            self.dataframe.to_parquet(filepath_feature_engineered, index=False, compression='snappy')
            print("Finish: Sanitization and feature engineering")
        return self.dataframe


    def calculate_momentum(self):
        print("step : Calculating momentum features")

        self.dataframe['log_return'] = np.log1p(self.dataframe['mom1m'])
        for i in tqdm(range(2, self.WINDOW + 1), desc="Calculating momentum features"):
            self.dataframe[f'log_mom{i}m'] = self.dataframe.groupby('permno')['log_return'].rolling(window=i).sum().reset_index(level=0, drop=True)
            self.dataframe[f'mom{i}m'] = np.expm1(self.dataframe[f'log_mom{i}m'])
        print(self.dataframe[['permno', 'mom48m']].tail(100))
        self.dataframe.drop(columns=[f'log_mom{i}m' for i in range(2, self.WINDOW + 1)], inplace=True)

        print("Finish: Calculating momentum features")
        

    def interpolate_with_median(self):
        print("step: Interpolating with median")

        numerical_columns = self.dataframe.select_dtypes(include=['float64', 'int64']).columns
        self.dataframe[numerical_columns] = self.dataframe.groupby('permno')[numerical_columns].transform(
            # lambda group: group.rolling(window=self.WINDOW, min_periods=1).median().fillna(method='bfill').fillna(method='ffill')
            lambda group: group.fillna(group.rolling(window=self.WINDOW, min_periods=1).median()).bfill()
        )
        print(self.dataframe[['permno', 'mom48m']].tail(100))

        print("Finish: Interpolating with median")





    def remove_high_nan_features(self):
        print("step: Removing high nan features")

        nan_percentages = self.dataframe.isna().mean()
        self.dataframe = self.dataframe.drop(columns=nan_percentages[nan_percentages >= self.NAN_THRESHOLD].index)
        self.dataframe = self.dataframe.ffill().bfill()
        print(self.dataframe[['permno', 'mom48m']].tail(100))

        print("Finish: Removing high nan features")


