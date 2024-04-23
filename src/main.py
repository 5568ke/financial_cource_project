# main.py
import pandas as pd
from data_processing.DataProcessor import DataProcessor
from clustering.kmeans import KMeansCluster
from clustering.dbscan import DBSCANCluster
#from clustering.agglomerative import AgglomerativeCluster
#from clustering.faiss import FaissCluster
from trading_strategy.trading import PairTradingService
from tqdm import tqdm as tdqm
import config
import os

def main():
    processor = DataProcessor(
        filepath='../datafile/datashare/datashare.csv',
        filepath_parquet='../datafile/datashare/datashare.parquet',
        min_year=1980,
        max_year=2021
    )
    processor.load_data()
    processor.preprocess_data()
    processor.sanitize_and_feature_engineer()
    pca_data = processor.reduce_dimensionality_with_pca()
    pca_data.columns = pca_data.columns.astype(str)
    print("Pca data : ",pca_data.tail(100))

    # Get corresponding file paths
    model_path = config.PATHS[config.CLUSTERING_ALGORITHM]['models']
    data_path = config.PATHS[config.CLUSTERING_ALGORITHM]['data']

    # Check if model and data files exist
    if os.path.exists(model_path) and os.path.exists(data_path):
        models_df = pd.read_pickle(model_path)
        clustered_data = pd.read_pickle(data_path)
        print("Loaded models and data from existing files.")
    else:
        print("No existing models and data from existing files.")
        # Select and apply the clustering algorithm
        if config.CLUSTERING_ALGORITHM == 'kmeans':
            model = KMeansCluster(**config.KMEANS_PARAMS)
        elif config.CLUSTERING_ALGORITHM == 'dbscan':
            model = DBSCANCluster(**config.DBSCAN_PARAMS)

        models_df, clustered_data = model.fit(pca_data)

        # Save model results
        models_df.to_pickle(model_path)
        clustered_data.to_pickle(data_path)
        print(f"Results saved to {model_path} and {data_path}")

    print("Cluster information:")
    print(models_df)
    print("Data information after clustering:")
    print(clustered_data['cluster'].value_counts())
    print("anootated data : ",clustered_data.tail(100))

    # process data for trading
    clustered_data['DATE'] = pd.to_datetime(clustered_data['DATE'])
    clustered_data.set_index('DATE', inplace=True)
    pair_trading_service = PairTradingService(std_dev_factor=1.5)

    # Group the data by month
    grouped_by_month = clustered_data.groupby(pd.Grouper(freq='ME'))

    month_data_dict = {}
    portfolios_dict = {}

    for month, month_data in tdqm(grouped_by_month):
        month_data_dict[month] = month_data
        grouped_by_cluster = month_data.groupby('cluster')
        current_month_portfolio = {'long': [], 'short': []}
        for cluster, cluster_data in grouped_by_cluster:
            pairs = pair_trading_service.find_pairs(cluster_data)
            portfolio = pair_trading_service.form_portfolio(pairs)
            current_month_portfolio['long'].extend(portfolio['long'])
            current_month_portfolio['short'].extend(portfolio['short'])

        portfolios_dict[month] = current_month_portfolio

    print("finish trading")

    # Calculate the returns for each monthly portfolio
    month_return = []
    for month in portfolios_dict:
        if month + pd.DateOffset(months=1) in month_data_dict:
            next_month_data = month_data_dict[month + pd.DateOffset(months=1)]
            portfolio = portfolios_dict[month]
            returns = pair_trading_service.calculate_monthly_returns(portfolio, next_month_data)
            print(f"Returns for {month.strftime('%Y-%m')}: {returns}%")
            month_return.append([month, returns])

    total_return = 1
    for data in month_return:
        total_return *= (1 + data[1])
    print("Total accumulated return percentage: ", total_return)


if __name__ == "__main__":
    main()
